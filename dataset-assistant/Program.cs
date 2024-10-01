// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.CommandLine;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using Microsoft.ML.Tokenizers;
using OpenAI;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using Spectre.Console;

namespace dataset_assistant;

public class Program
{
    public const int Maxtokens = 8192;

    [Experimental("OPENAI001")]
    private static async Task<int> Main(string[] args)
    {
        #region Options

        var dumpPathOption = new Option<string>(
            ["--dbDumpPath", "-D"],
            "The path to the database dump file. This file should be a csv file"
        )
        {
            IsRequired = true
        };

        var embeddingsModelOption = new Option<string>(
            ["--embeddingsModel", "-e"],
            () => "text-embedding-3-small",
            "The embeddings model to use"
        );

        var openAiApiKeyOption = new Option<string>(
            ["--openAiApiKey", "-k"],
            "The OpenAI API key"
        );

        var qdrantUrlOption = new Option<string>(
            ["--qdrantUrl", "-q"],
            () => "localhost:6334",
            "The URL of the Qdrant server"
        );

        var batchOption = new Option<bool>(
            "--batch",
            "Use batching"
        );

        var dryrunOption = new Option<bool>(
            "--dryrun",
            "Don't touch QDrant, just do the processing. Also skips embedding, adding artificial delays to simulate the process"
        );

        #endregion

        var rootCommand = new RootCommand("GalaxyGPT Dataset Management Assistant")
        {
            qdrantUrlOption,
            dumpPathOption,
            embeddingsModelOption,
            openAiApiKeyOption,
            batchOption,
            dryrunOption
        };

        rootCommand.SetHandler(async handler =>
        {
            #region Option Values

            string qdrantUrlOptionValue = handler.ParseResult.GetValueForOption(qdrantUrlOption)!;
            string dumpPathOptionValue = handler.ParseResult.GetValueForOption(dumpPathOption)!;
            string embeddingsModelOptionValue = handler.ParseResult.GetValueForOption(embeddingsModelOption)!;
            string? openAiApiKeyOptionValue = handler.ParseResult.GetValueForOption(openAiApiKeyOption);
            bool batchOptionValue = handler.ParseResult.GetValueForOption(batchOption);
            bool dryrunOptionValue = handler.ParseResult.GetValueForOption(dryrunOption);

            #endregion

            // Throwing all the logic into a callback makes me want to cry because it's so bad code quality wise, but there's literally no other way
            // Probably a good idea to extract some of the logic into seperate methods in the future, probably when we start working on ADCS
            await AnsiConsole.Progress().Columns(new SpinnerColumn(), new TaskDescriptionColumn(),
                new ProgressBarColumn(), new PercentageColumn(), new RemainingTimeColumn()).StartAsync(async ctx =>
            {
                #region Tasks

                ProgressTask deptask = ctx.AddTask("Setting up dependencies");
                ProgressTask dbdumpTask = ctx.AddTaskAfter("Reading Database Dump", deptask);
                ProgressTask chunkingTask = ctx.AddTaskAfter("Chunking Pages", dbdumpTask);
                ProgressTask embeddingTask = ctx.AddTaskAfter("Generating Embeddings", chunkingTask);
                ProgressTask? upsertTask = !dryrunOptionValue ? ctx.AddTaskAfter("Upserting Points into QDrant", embeddingTask) : null;

                #endregion

                #region Dependencies

                var embeddingsTokenizer = TiktokenTokenizer.CreateForModel(embeddingsModelOptionValue);
                OpenAIClient openAiClient = new(openAiApiKeyOptionValue ??
                                                Environment.GetEnvironmentVariable("OPENAI_API_KEY") ??
                                                throw new InvalidOperationException());

                string[] qdrantUrlAndPort = qdrantUrlOptionValue.Split(':');
                var qdrantClient = new QdrantClient(qdrantUrlAndPort[0], qdrantUrlAndPort.Length == 2
                    ? int.Parse(qdrantUrlAndPort[1])
                    : 6334);

                deptask.Value(100);

                #endregion

                List<(string title, string content)> pages = await DatasetCreator.GetPagesFromCsv(dumpPathOptionValue);
                dbdumpTask.Value(100);
                dbdumpTask.StopTask();

                List<(string title, string content, int tokencount)> chunksList = DatasetCreator.ChunkPages(pages, chunkingTask, embeddingsTokenizer);

                List<(string title, string content, int tokenscount, float[] embeddings)> embeddedChunks;

                if (batchOptionValue)
                {
                    embeddingTask.MaxValue(100);
                    embeddedChunks = await BatchRequestModule.CreateAndProcessBatchRequest(chunksList, embeddingsModelOptionValue, openAiClient.GetFileClient(), openAiClient.GetBatchClient(), embeddingTask);
                }
                else
                {
                    embeddingTask.MaxValue(chunksList.Count);
                    embeddedChunks =
                        await DatasetCreator.GenerateEmbeddedChunks(chunksList, embeddingTask, openAiClient, embeddingsModelOptionValue);
                }

                if (!dryrunOptionValue)
                {
                    await qdrantClient.RecreateCollectionAsync("galaxypedia", new VectorParams
                    {
                        Distance = Distance.Cosine,
                        Size = 1536
                    });
                    await DatasetCreator.UpsertPointsIntoQdrant(embeddedChunks, qdrantClient);
                    upsertTask?.Value(100);
                }
            });
        });

        var dumpCommand = new Command("dump", "Dump the database to the output directory");
        rootCommand.AddCommand(dumpCommand);

        dumpCommand.SetHandler(async _ =>
        {
            await using Stream? stream = Assembly.GetExecutingAssembly().GetManifestResourceStream("dataset_assistant.dump-database.sh");

            if (stream == null)
                throw new InvalidOperationException("Failed to get the dump.sh resource");

            using StreamReader reader = new(stream);
            string script = await reader.ReadToEndAsync();

            try
            {
                Console.WriteLine("Extracting the dump script");
                // Write the script to /tmp/dump.sh
                await File.WriteAllTextAsync("/tmp/dump.sh", script);

                // Make the script executable
                using Process? processthing = Process.Start(new ProcessStartInfo
                {
                    FileName = "chmod",
                    Arguments = "+x /tmp/dump.sh",
                    UseShellExecute = true
                });

                if (processthing == null)
                    throw new InvalidOperationException("Failed to start the chmod process");

                await processthing.WaitForExitAsync();

                // run the script
                using Process? process = Process.Start(new ProcessStartInfo
                {
                    FileName = "/tmp/dump.sh",
                    UseShellExecute = true
                });

                if (process == null)
                    throw new InvalidOperationException("Failed to start the dump.sh process");

                await process.WaitForExitAsync();

                Console.WriteLine("Database dumped successfully");
            }
            finally
            {
                File.Delete("/tmp/dump.sh");
                Console.WriteLine("Deleting the dump script");
            }
        });

        return await rootCommand.InvokeAsync(args);
    }
}