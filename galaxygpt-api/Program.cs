// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Diagnostics;
using System.Reflection;
using Asp.Versioning.ApiExplorer;
using Asp.Versioning.Builder;
using galaxygpt;
using Microsoft.Extensions.Options;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Embeddings;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace galaxygpt_api;

public class Program
{
    public static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        builder.Logging.AddConsole();
        builder.Services.AddProblemDetails();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddApiVersioning(options => { options.ReportApiVersions = true; }).AddApiExplorer(options =>
        {
            options.GroupNameFormat = "'v'VVV";
            options.SubstituteApiVersionInUrl = true;
        }).EnableApiVersionBinding();

        builder.Services.AddTransient<IConfigureOptions<SwaggerGenOptions>, ConfigureSwaggerOptions>();
        builder.Services.AddSwaggerGen(options =>
        {
            options.OperationFilter<SwaggerDefaultValues>();
            options.IncludeXmlComments(Path.Combine(AppContext.BaseDirectory, $"{Assembly.GetExecutingAssembly().GetName().Name}.xml"));
        });
        builder.Services.AddMemoryCache();

        #region Configuration

        IConfigurationRoot configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", true, true)
            .AddEnvironmentVariables()
            .AddUserSecrets<Program>()
            .Build();

        builder.Configuration.Sources.Clear();
        builder.Configuration.AddConfiguration(configuration);

        #endregion

        builder.WebHost.UseSentry(o =>
        {
            o.Dsn = configuration["SENTRY_DSN"] ??
                    "https://1df72bed08400836796f15c03748d195@o4507833886834688.ingest.us.sentry.io/4507833934544896";
#if DEBUG
            o.Debug = true;
#endif
            o.TracesSampleRate = 1.0;
            o.ProfilesSampleRate = 1.0;
        });

        #region GalaxyGPT Services

        var openAiClient = new OpenAIClient(configuration["OPENAI_API_KEY"] ?? throw new InvalidOperationException("No OpenAI API key was provided."));
        string gptModel = configuration["GPT_MODEL"] ?? "gpt-4o-mini";
        string textEmbeddingModel = configuration["TEXT_EMBEDDING_MODEL"] ?? "text-embedding-3-small";
        string moderationModel = configuration["MODERATION_MODEL"] ?? "text-moderation-stable";

        builder.Services.AddSingleton(openAiClient.GetChatClient(gptModel));
        builder.Services.AddSingleton(openAiClient.GetEmbeddingClient(textEmbeddingModel));
        builder.Services.AddSingleton(openAiClient.GetModerationClient(moderationModel));
        builder.Services.AddKeyedSingleton("gptTokenizer", TiktokenTokenizer.CreateForModel(gptModel));
        builder.Services.AddKeyedSingleton("embeddingsTokenizer", TiktokenTokenizer.CreateForModel(textEmbeddingModel));
        builder.Services.AddSingleton(provider => new ContextManager(
            provider.GetRequiredService<EmbeddingClient>(),
            provider.GetRequiredKeyedService<TiktokenTokenizer>("embeddingsTokenizer"),
            configuration["QDRANT_URL"]
        ));
        builder.Services.AddSingleton<AiClient>();

        #endregion

        WebApplication app = builder.Build();
        IVersionedEndpointRouteBuilder versionedApi = app.NewVersionedApi("galaxygpt");
        IVersionedEndpointRouteBuilder adcsApi = app.NewVersionedApi("adcs");

        app.UseHttpsRedirection();
        app.UseExceptionHandler(exceptionHandlerApp =>
            exceptionHandlerApp.Run(async context => await Results.Problem().ExecuteAsync(context)));

        var galaxyGpt = app.Services.GetRequiredService<AiClient>();
        var contextManager = app.Services.GetRequiredService<ContextManager>();

        #region API

        RouteGroupBuilder v1 = versionedApi.MapGroup("/api/v{version:apiVersion}").HasApiVersion(1.0);

        v1.MapPost("ask", async (AskPayload askPayload) =>
        {
            if (string.IsNullOrWhiteSpace(askPayload.Prompt))
                return Results.BadRequest("The question cannot be empty.");

            var requestStart = Stopwatch.StartNew();

            (string, int) context = await contextManager.FetchContext(askPayload.Prompt);
            string answer = await galaxyGpt.AnswerQuestion(askPayload.Prompt, context.Item1, username: askPayload.Username);

            requestStart.Stop();

            var results = new AskResponse
            {
                Answer = answer.Trim(),
                Context = context.Item1,
                Duration = requestStart.ElapsedMilliseconds.ToString(),
                Version = Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? string.Empty
            };

            return Results.Json(results);
        }).WithName("AskQuestion").WithOpenApi().Produces<AskResponse>();

        #endregion

        #region ADCS

        RouteGroupBuilder adcsGroup = adcsApi.MapGroup("/api/v{version:apiVersion}/adcs").HasApiVersion(1.0);
        adcsGroup.MapPost("start", new Func<object>(() => throw new NotImplementedException()));

        adcsGroup.MapPost("stop", new Func<object>(() => throw new NotImplementedException()));

        adcsGroup.MapPost("force-create", new Func<object>(() => throw new NotImplementedException()));

        #endregion

        app.UseSwagger();
        if (app.Environment.IsDevelopment())
        {
            app.UseSwaggerUI(options =>
            {
                foreach ( ApiVersionDescription description in app.DescribeApiVersions() )
                {
                    options.SwaggerEndpoint(
                        $"/swagger/{description.GroupName}/swagger.json",
                        description.GroupName );
                }
            });
        }
        app.Run();
    }
}

public class AskPayload
{
    /// <summary>
    /// The question to ask the AI
    /// </summary>
    /// <example>What is the deity?</example>
    [Required]
    public required string Prompt { get; init; }

    /// <summary>
    /// The model to use for the request
    /// </summary>
    [DefaultValue("gpt-4o-mini")]
    public string? Model { get; init; } = "gpt-4o-mini";

    /// <summary>
    /// The username of the user asking the question
    /// </summary>
    public string? Username { get; init; }

    /// <summary>
    /// The maximum amount of tokens to generate
    /// </summary>
    public int? MaxLength { get; init; }
}

public class AskResponse
{
    public required string Answer { get; init; }
    public required string Context { get; init; }
    public required string Duration { get; init; }
    public required string Version { get; init; }
}