using Asp.Versioning.ApiExplorer;
using Microsoft.Extensions.Options;
using Microsoft.OpenApi.Models;
using Swashbuckle.AspNetCore.SwaggerGen;

namespace galaxygpt_api;

public class ConfigureSwaggerOptions(IApiVersionDescriptionProvider provider) : IConfigureOptions<SwaggerGenOptions>
{
    public void Configure( SwaggerGenOptions options )
    {
        foreach ( ApiVersionDescription description in provider.ApiVersionDescriptions )
        {
            options.SwaggerDoc(
                description.GroupName,
                new OpenApiInfo
                {
                    Title = "GalaxyGPT API",
                    Version = description.ApiVersion.ToString(),
                } );
        }
    }
}