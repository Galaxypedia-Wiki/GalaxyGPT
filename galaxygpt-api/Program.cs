// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.Reflection;
using Asp.Versioning.Builder;
using galaxygpt;
using galaxygpt.Database;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.Extensions.Options;
using Microsoft.ML.Tokenizers;
using OpenAI;
using Sentry.Profiling;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace galaxygpt_api;

public class Program
{
    public static void Main(string[] args)
    {
        WebApplicationBuilder builder = WebApplication.CreateBuilder(args);

        // Add services to the container.
        builder.Logging.AddConsole();
        builder.Services.AddProblemDetails();
        builder.Services.AddEndpointsApiExplorer();
        builder.Services.AddApiVersioning(options =>
        {
            options.ReportApiVersions = true;
        }).AddApiExplorer(options =>
        {
            options.GroupNameFormat = "'v'VVV";
            options.SubstituteApiVersionInUrl = true;
        }).EnableApiVersionBinding();

        builder.Services.AddTransient<IConfigureOptions<SwaggerGenOptions>, ConfigureSwaggerOptions>();
        builder.Services.AddSwaggerGen(options => options.OperationFilter<SwaggerDefaultValues>());
        builder.Services.AddMemoryCache();

        #region Configuration

        IConfigurationRoot configuration = new ConfigurationBuilder()
            .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
            .AddEnvironmentVariables()
            .AddUserSecrets<Program>()
            .Build();

        builder.Configuration.Sources.Clear();
        builder.Configuration.AddConfiguration(configuration);

        #endregion

        builder.WebHost.UseSentry(o =>
        {
            o.Dsn = configuration["SENTRY_DSN"] ?? "https://1df72bed08400836796f15c03748d195@o4507833886834688.ingest.us.sentry.io/4507833934544896";
#if DEBUG
            o.Debug = true;
#endif
            o.TracesSampleRate = 1.0;
            o.ProfilesSampleRate = 1.0;
        });

        #region GalaxyGPT Services

        var openAiClient = new OpenAIClient(configuration["OPENAI_API_KEY"] ?? throw new InvalidOperationException());
        string gptModel = configuration["GPT_MODEL"] ?? "gpt-4o-mini";
        string textEmbeddingModel = configuration["TEXT_EMBEDDING_MODEL"] ?? "text-embedding-3-small";
        string moderationModel = configuration["MODERATION_MODEL"] ?? "text-moderation-stable";

        builder.Services.AddSingleton(new VectorDb());
        builder.Services.AddSingleton(openAiClient.GetChatClient(gptModel));
        builder.Services.AddSingleton(openAiClient.GetEmbeddingClient(textEmbeddingModel));
        builder.Services.AddSingleton(openAiClient.GetModerationClient(moderationModel));
        builder.Services.AddKeyedSingleton("gptTokenizer", TiktokenTokenizer.CreateForModel("gpt-4o-mini"));
        builder.Services.AddKeyedSingleton("embeddingsTokenizer", TiktokenTokenizer.CreateForModel("text-embedding-3-small"));
        builder.Services.AddSingleton<ContextManager>();
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

            (string, int) context = await contextManager.FetchContext(askPayload.Prompt);
            string answer = await galaxyGpt.AnswerQuestion(askPayload.Prompt, context.Item1, 4096, username: askPayload.Username);

            var results = new Dictionary<string, string>
            {
                { "answer", answer.Trim() },
                { "context", context.Item1 },
                { "version", Assembly.GetExecutingAssembly().GetName().Version?.ToString() ?? string.Empty}
            };

            return Results.Json(results);
        }).WithName("AskQuestion").WithOpenApi();

        #endregion

        #region ADCS

        RouteGroupBuilder adcsGroup = adcsApi.MapGroup("/api/v{version:apiVersion}/adcs").HasApiVersion(1.0);
        adcsGroup.MapPost("start", new Func<object>(() => throw new NotImplementedException()));

        adcsGroup.MapPost("stop", new Func<object>(() => throw new NotImplementedException()));

        adcsGroup.MapPost("force-create", new Func<object>(() => throw new NotImplementedException()));

        #endregion

        app.UseSwagger();
        // if (app.Environment.IsDevelopment())
        // {
        //     app.UseSwaggerUI(options =>
        //     {
        //         foreach ( ApiVersionDescription description in app.DescribeApiVersions() )
        //         {
        //             options.SwaggerEndpoint(
        //                 $"/swagger/{description.GroupName}/swagger.json",
        //                 description.GroupName );
        //         }
        //     });
        // }
        app.Run();
    }
}

// ReSharper disable once ArrangeTypeModifiers
// ReSharper disable once ClassNeverInstantiated.Global
class AskPayload
{
    public required string Prompt { get; set; }
    public string? Model { get; set; }
    public string? Username { get; set; }
}