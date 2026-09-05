using System.Diagnostics;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;

var builder = WebApplication.CreateBuilder(args);

builder.WebHost.UseUrls("http://localhost:5080");
builder.Services.AddHttpClient("ollama", c =>
{
    c.BaseAddress = new Uri("http://127.0.0.1:11434");
    c.Timeout = TimeSpan.FromMinutes(10);
});

var app = builder.Build();

var dataDir = Directory.GetCurrentDirectory();
var nameStore = new NameStore(dataDir, AppContext.BaseDirectory);
var nameFile = nameStore.Path;
var uploadsDir = Path.Combine(dataDir, "uploads");
Directory.CreateDirectory(uploadsDir);

nameStore.EnsureFile();
await OllamaHelper.EnsureRunningAsync();

app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/status", async (IHttpClientFactory factory) =>
{
    var running = await OllamaHelper.IsRunningAsync(factory);
    return Results.Json(new
    {
        ollama = running,
        user = nameStore.Read(),
        hasName = nameStore.HasName()
    });
});

app.MapGet("/api/user", () =>
{
    var name = nameStore.Read();
    return Results.Json(new
    {
        name,
        hasName = !string.IsNullOrWhiteSpace(name)
    });
});

app.MapPost("/api/user", async (HttpRequest request) =>
{
    using var doc = await JsonDocument.ParseAsync(request.Body);
    var name = doc.RootElement.TryGetProperty("name", out var n)
        ? n.GetString()?.Trim() ?? ""
        : "";

    if (name.Length > 60)
        name = name[..60];

    await nameStore.WriteAsync(name);
    return Results.Json(new
    {
        name,
        hasName = !string.IsNullOrWhiteSpace(name)
    });
});

app.MapDelete("/api/user", async () =>
{
    await nameStore.WriteAsync("");
    return Results.Json(new { name = "", hasName = false });
});

app.MapGet("/api/models", async (IHttpClientFactory factory) =>
{
    var client = factory.CreateClient("ollama");
    try
    {
        await OllamaHelper.EnsureRunningAsync();
        using var response = await client.GetAsync("/api/tags");
        if (!response.IsSuccessStatusCode)
            return Results.Json(new { models = Array.Empty<object>(), error = "Ollama не отвечает" });

        using var stream = await response.Content.ReadAsStreamAsync();
        using var doc = await JsonDocument.ParseAsync(stream);
        var models = new List<object>();

        if (doc.RootElement.TryGetProperty("models", out var arr) && arr.ValueKind == JsonValueKind.Array)
        {
            foreach (var item in arr.EnumerateArray())
            {
                var name = item.TryGetProperty("name", out var nm) ? nm.GetString() : null;
                if (string.IsNullOrWhiteSpace(name)) continue;
                models.Add(new
                {
                    name,
                    size = item.TryGetProperty("size", out var sz) ? sz.GetInt64() : 0,
                    modified = item.TryGetProperty("modified_at", out var md) ? md.GetString() : ""
                });
            }
        }

        return Results.Json(new { models });
    }
    catch (Exception ex)
    {
        return Results.Json(new { models = Array.Empty<object>(), error = ex.Message });
    }
});

app.MapPost("/api/upload", async (HttpRequest request) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "Ожидается multipart/form-data" });

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("file") ?? form.Files.FirstOrDefault();
    if (file is null || file.Length == 0)
        return Results.BadRequest(new { error = "Файл не выбран" });

    if (file.Length > 20 * 1024 * 1024)
        return Results.BadRequest(new { error = "Файл больше 20 МБ" });

    var safeName = AppUtil.SanitizeFileName(file.FileName);
    var savedPath = Path.Combine(uploadsDir, $"{DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()}_{safeName}");
    await using (var fs = File.Create(savedPath))
        await file.CopyToAsync(fs);

    var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
    var mime = string.IsNullOrWhiteSpace(file.ContentType) ? AppUtil.GuessMime(ext) : file.ContentType;
    var kind = AppUtil.ClassifyFile(ext, mime);

    string? textContent = null;
    string? base64 = null;
    string preview = "";

    switch (kind)
    {
        case "image":
            var bytes = await File.ReadAllBytesAsync(savedPath);
            base64 = Convert.ToBase64String(bytes);
            preview = $"[изображение: {file.FileName}]";
            break;
        case "text":
            textContent = await AppUtil.ReadTextFileAsync(savedPath);
            if (textContent.Length > 80_000)
                textContent = textContent[..80_000] + "\n\n[файл обрезан]";
            preview = $"[текст: {file.FileName}, {textContent.Length} символов]";
            break;
        case "pdf":
            textContent = PdfText.Extract(savedPath);
            if (string.IsNullOrWhiteSpace(textContent))
                textContent = $"(Не удалось извлечь текст из PDF «{file.FileName}». Файл сохранён на сервере.)";
            else if (textContent.Length > 80_000)
                textContent = textContent[..80_000] + "\n\n[PDF обрезан]";
            preview = $"[PDF: {file.FileName}]";
            break;
        default:
            preview = $"[файл: {file.FileName} ({AppUtil.FormatSize(file.Length)})]";
            textContent = $"Пользователь приложил файл «{file.FileName}» ({mime}, {AppUtil.FormatSize(file.Length)}). Содержимое бинарное и не прочитано как текст.";
            break;
    }

    return Results.Json(new
    {
        fileName = file.FileName,
        savedAs = Path.GetFileName(savedPath),
        kind,
        mime,
        size = file.Length,
        content = textContent,
        imageBase64 = base64,
        preview
    });
});

app.MapPost("/api/chat", async (HttpRequest request, IHttpClientFactory factory) =>
{
    ChatRequest? body;
    try
    {
        body = await JsonSerializer.DeserializeAsync<ChatRequest>(request.Body, AppJson.Options);
    }
    catch
    {
        return Results.BadRequest(new { error = "Некорректный JSON" });
    }

    if (body is null || string.IsNullOrWhiteSpace(body.Model))
        return Results.BadRequest(new { error = "Не выбрана модель" });

    if (body.Messages is null || body.Messages.Count == 0)
        return Results.BadRequest(new { error = "Нет сообщений" });

    var userName = nameStore.Read();
    var outgoing = new List<object>();

    outgoing.Add(new
    {
        role = "system",
        content = AppUtil.BuildSystemPrompt(userName)
    });

    foreach (var msg in body.Messages)
    {
        if (string.IsNullOrWhiteSpace(msg.Role)) continue;
        var role = msg.Role.Trim().ToLowerInvariant();
        if (role is not ("user" or "assistant" or "system")) continue;

        if (msg.Images is { Count: > 0 })
        {
            outgoing.Add(new
            {
                role,
                content = msg.Content ?? "",
                images = msg.Images
            });
        }
        else
        {
            outgoing.Add(new
            {
                role,
                content = msg.Content ?? ""
            });
        }
    }

    var client = factory.CreateClient("ollama");
    try
    {
        await OllamaHelper.EnsureRunningAsync();
        using var response = await client.PostAsJsonAsync("/api/chat", new
        {
            model = body.Model,
            messages = outgoing,
            stream = false
        });

        var raw = await response.Content.ReadAsStringAsync();
        if (!response.IsSuccessStatusCode)
        {
            return Results.Json(new
            {
                error = $"Ollama вернула ошибку {(int)response.StatusCode}",
                details = AppUtil.Trim(raw, 800)
            }, statusCode: 502);
        }

        using var doc = JsonDocument.Parse(raw);
        var content = "";
        if (doc.RootElement.TryGetProperty("message", out var message) &&
            message.TryGetProperty("content", out var c))
        {
            content = c.GetString() ?? "";
        }

        return Results.Json(new { content, model = body.Model });
    }
    catch (TaskCanceledException)
    {
        return Results.Json(new { error = "Модель слишком долго отвечает. Попробуйте ещё раз." }, statusCode: 504);
    }
    catch (HttpRequestException)
    {
        return Results.Json(new { error = "Не удалось связаться с Ollama. Проверьте, что приложение запустило ollama serve." }, statusCode: 503);
    }
    catch (Exception ex)
    {
        return Results.Json(new { error = ex.Message }, statusCode: 500);
    }
});

app.MapFallbackToFile("index.html");

Console.WriteLine();
Console.WriteLine("  Нейросети · личный помощник");
Console.WriteLine("  Сайт:   http://localhost:5080");
Console.WriteLine("  Ollama: http://127.0.0.1:11434");
Console.WriteLine($"  Имя:    {nameFile}");
Console.WriteLine();

app.Run();

sealed class NameStore
{
    public string Path { get; }

    public NameStore(params string[] directories)
    {
        foreach (var dir in directories)
        {
            if (string.IsNullOrWhiteSpace(dir)) continue;
            var candidate = System.IO.Path.Combine(dir, "name.json");
            if (File.Exists(candidate))
            {
                Path = candidate;
                return;
            }
        }

        var root = directories.FirstOrDefault(d => !string.IsNullOrWhiteSpace(d))
                   ?? Directory.GetCurrentDirectory();
        Path = System.IO.Path.Combine(root, "name.json");
    }

    public void EnsureFile()
    {
        if (File.Exists(Path)) return;
        File.WriteAllText(Path, """
            {
              "name": ""
            }
            """, Encoding.UTF8);
    }

    public bool HasName() => !string.IsNullOrWhiteSpace(Read());

    public string Read()
    {
        try
        {
            if (!File.Exists(Path)) return "";
            var raw = File.ReadAllText(Path).Trim();
            if (raw.Length == 0 || raw == "{}") return "";

            using var doc = JsonDocument.Parse(raw);
            if (doc.RootElement.ValueKind == JsonValueKind.String)
                return doc.RootElement.GetString()?.Trim() ?? "";

            if (doc.RootElement.TryGetProperty("name", out var n))
            {
                if (n.ValueKind == JsonValueKind.Null) return "";
                return n.GetString()?.Trim() ?? "";
            }
        }
        catch { }

        return "";
    }

    public Task WriteAsync(string name)
    {
        EnsureFile();
        var payload = JsonSerializer.Serialize(
            new { name = name ?? "" },
            new JsonSerializerOptions { WriteIndented = true });
        return File.WriteAllTextAsync(Path, payload, Encoding.UTF8);
    }
}

static class AppUtil
{
    public static string BuildSystemPrompt(string userName)
    {
        var sb = new StringBuilder();
        sb.Append("Ты вежливый и полезный личный помощник. Отвечай на языке пользователя. ");
        sb.Append("Ответы держи ясными и аккуратными.");
        if (!string.IsNullOrWhiteSpace(userName))
        {
            sb.Append(" Пользователя зовут ").Append(userName).Append('.');
            sb.Append(" Когда уместно, обращайся к нему по имени.");
        }
        return sb.ToString();
    }

    public static string SanitizeFileName(string name)
    {
        var just = Path.GetFileName(name);
        foreach (var c in Path.GetInvalidFileNameChars())
            just = just.Replace(c, '_');
        return string.IsNullOrWhiteSpace(just) ? "file.bin" : just;
    }

    public static string GuessMime(string ext) => ext switch
    {
        ".png" => "image/png",
        ".jpg" or ".jpeg" => "image/jpeg",
        ".gif" => "image/gif",
        ".webp" => "image/webp",
        ".bmp" => "image/bmp",
        ".pdf" => "application/pdf",
        ".html" or ".htm" => "text/html",
        ".txt" => "text/plain",
        ".md" => "text/markdown",
        ".json" => "application/json",
        ".csv" => "text/csv",
        ".xml" => "application/xml",
        ".css" => "text/css",
        ".js" => "text/javascript",
        ".cs" => "text/plain",
        _ => "application/octet-stream"
    };

    public static string ClassifyFile(string ext, string mime)
    {
        if (ext is ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp" || mime.StartsWith("image/"))
            return "image";
        if (ext is ".pdf" || mime.Contains("pdf"))
            return "pdf";
        if (ext is ".txt" or ".md" or ".html" or ".htm" or ".json" or ".csv" or ".xml" or ".css" or ".js"
            or ".cs" or ".py" or ".ts" or ".tsx" or ".jsx" or ".yml" or ".yaml" or ".log" or ".ini" or ".cfg"
            or ".svg" or ".rtf" or ".tex" or ".sql" or ".sh" or ".bat" or ".ps1"
            || mime.StartsWith("text/") || mime.Contains("json") || mime.Contains("xml"))
            return "text";
        return "binary";
    }

    public static async Task<string> ReadTextFileAsync(string path)
    {
        var bytes = await File.ReadAllBytesAsync(path);
        if (bytes.Length >= 3 && bytes[0] == 0xEF && bytes[1] == 0xBB && bytes[2] == 0xBF)
            return Encoding.UTF8.GetString(bytes, 3, bytes.Length - 3);
        if (bytes.Length >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFE)
            return Encoding.Unicode.GetString(bytes);
        if (bytes.Length >= 2 && bytes[0] == 0xFE && bytes[1] == 0xFF)
            return Encoding.BigEndianUnicode.GetString(bytes);
        return Encoding.UTF8.GetString(bytes);
    }

    public static string FormatSize(long bytes)
    {
        if (bytes < 1024) return $"{bytes} Б";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:0.#} КБ";
        return $"{bytes / (1024.0 * 1024.0):0.#} МБ";
    }

    public static string Trim(string s, int max) => s.Length <= max ? s : s[..max] + "…";
}

static class AppJson
{
    public static readonly JsonSerializerOptions Options = new()
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };
}

sealed class ChatRequest
{
    public string? Model { get; set; }
    public List<ChatMessageDto>? Messages { get; set; }
}

sealed class ChatMessageDto
{
    public string? Role { get; set; }
    public string? Content { get; set; }
    public List<string>? Images { get; set; }
}

static class OllamaHelper
{
    static readonly object Gate = new();
    static Process? _process;
    static DateTime _lastAttempt = DateTime.MinValue;

    public static async Task<bool> IsRunningAsync(IHttpClientFactory? factory = null)
    {
        try
        {
            using var client = factory?.CreateClient("ollama") ?? new HttpClient { Timeout = TimeSpan.FromSeconds(2) };
            if (factory is null)
                client.BaseAddress = new Uri("http://127.0.0.1:11434");
            using var response = await client.GetAsync("/api/tags");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }

    public static async Task EnsureRunningAsync()
    {
        if (await IsRunningAsync()) return;

        lock (Gate)
        {
            if (DateTime.UtcNow - _lastAttempt < TimeSpan.FromSeconds(4))
                return;
            _lastAttempt = DateTime.UtcNow;

            try
            {
                if (_process is { HasExited: false })
                    return;

                var psi = new ProcessStartInfo
                {
                    FileName = "ollama",
                    Arguments = "serve",
                    UseShellExecute = false,
                    CreateNoWindow = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true
                };

                _process = Process.Start(psi);
                Console.WriteLine("  Запущен процесс: ollama serve");
            }
            catch (Exception ex)
            {
                Console.WriteLine("  Не удалось запустить ollama serve: " + ex.Message);
            }
        }

        for (var i = 0; i < 20; i++)
        {
            await Task.Delay(250);
            if (await IsRunningAsync()) return;
        }
    }
}

static class PdfText
{
    public static string Extract(string path)
    {
        try
        {
            var raw = File.ReadAllText(path, Encoding.Latin1);
            var sb = new StringBuilder();

            foreach (Match m in Regex.Matches(raw, @"BT(.*?)ET", RegexOptions.Singleline))
            {
                foreach (Match t in Regex.Matches(m.Groups[1].Value, @"\((?:\\.|[^\\)])*\)"))
                {
                    var piece = t.Value[1..^1];
                    piece = piece
                        .Replace("\\n", "\n")
                        .Replace("\\r", "\n")
                        .Replace("\\t", "\t")
                        .Replace("\\(", "(")
                        .Replace("\\)", ")")
                        .Replace("\\\\", "\\");
                    sb.Append(piece);
                    if (!piece.EndsWith(' ') && !piece.EndsWith('\n'))
                        sb.Append(' ');
                }
                sb.AppendLine();
            }

            var text = Regex.Replace(sb.ToString(), "[ \t]+", " ");
            text = Regex.Replace(text, @"\n{3,}", "\n\n").Trim();
            return text;
        }
        catch
        {
            return "";
        }
    }
}
