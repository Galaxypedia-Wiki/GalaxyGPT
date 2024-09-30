FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS base
USER $APP_UID
WORKDIR /app
EXPOSE 3636

FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
ARG BUILD_CONFIGURATION=Release
WORKDIR /src
COPY ["galaxygpt-api/galaxygpt-api.csproj", "galaxygpt-api/"]
COPY ["galaxygpt/galaxygpt.csproj", "galaxygpt/"]
RUN dotnet restore "galaxygpt-api/galaxygpt-api.csproj"
COPY .git .git
COPY . .
WORKDIR "/src/galaxygpt-api"
RUN dotnet build "galaxygpt-api.csproj" -c $BUILD_CONFIGURATION -o /app/build

FROM build AS publish
ARG BUILD_CONFIGURATION=Release
RUN dotnet publish "galaxygpt-api.csproj" -c $BUILD_CONFIGURATION -o /app/publish /p:UseAppHost=false

FROM base AS final
WORKDIR /app
COPY --from=publish /app/publish .
ENTRYPOINT ["dotnet", "galaxygpt-api.dll"]
