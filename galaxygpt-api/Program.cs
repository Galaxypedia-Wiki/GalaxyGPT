
using System.Reflection;
using Asp.Versioning.Builder;
using galaxygpt;
using Microsoft.Extensions.Options;
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

        builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);

        WebApplication app = builder.Build();
        IVersionedEndpointRouteBuilder versionedApi = app.NewVersionedApi("galaxygpt");
        IVersionedEndpointRouteBuilder adcsApi = app.NewVersionedApi("adcs");

        app.UseHttpsRedirection();

        app.UseExceptionHandler(exceptionHandlerApp =>
            exceptionHandlerApp.Run(async context => await Results.Problem().ExecuteAsync(context)));

        #region API
        RouteGroupBuilder v1 = versionedApi.MapGroup("/api/v{version:apiVersion}").HasApiVersion(1.0);

        v1.MapPost("ask", async (AskPayload askPayload) =>
        {
            if (string.IsNullOrEmpty(askPayload.Prompt))
                return Results.BadRequest("The question cannot be empty.");

            (string, int) context = await GalaxyGpt.FetchContext(askPayload.Prompt, "text-embedding-3-small");
            string answer = await GalaxyGpt.AnswerQuestion(askPayload.Prompt, context.Item1, askPayload.Model ?? app.Configuration["MODEL"] ?? throw new InvalidOperationException(), 4096, 4096, username: askPayload.Username);

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