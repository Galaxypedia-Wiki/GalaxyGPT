// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using dataset_assistant;
using Microsoft.ML.Tokenizers;
using OpenAI;
using Qdrant.Client;
using Qdrant.Client.Grpc;
using Quartz;

namespace ADCS;

public class UpdateDataSetJob : IJob
{
    // Create a new dataset from a database dump
    // Some things to keep in mind:
    // - We should use the batch api
    // - We should run diffing. Only update content that has changed. This saves costs and time.
    // - Architecture should be similar to what is done in dataset-assistant, but without deleting and recreating the collection

    public async Task Execute(IJobExecutionContext context)
    {
        // Get some dependencies
        OpenAIClient openAiClient = context.JobDetail.JobDataMap.Get("openAiClient") as OpenAIClient ?? throw new InvalidOperationException();
        QdrantClient qdrantClient = context.JobDetail.JobDataMap.Get("qdrantClient") as QdrantClient ?? throw new InvalidOperationException();
        string dbDumpPath = context.JobDetail.JobDataMap.GetString("dbDumpPath") ?? throw new InvalidOperationException();
        string embeddingsModel = context.JobDetail.JobDataMap.GetString("embeddingsModel") ?? throw new InvalidOperationException();
        var embeddingsTokenizer = TiktokenTokenizer.CreateForModel(embeddingsModel);

        // Start by verifying that the collection exists
        if (!await qdrantClient.CollectionExistsAsync("galaxypedia"))
        {
            await qdrantClient.CreateCollectionAsync("galaxypedia", new VectorParams
            {
                Distance = Distance.Cosine,
                Size = 1536
            });
        }

        List<(string title, string content)> pages = await DatasetCreator.GetPagesFromCsv(dbDumpPath);

        // Grab the existing pages from Qdrant
        IReadOnlyList<ScoredPoint> qdrantCollection = await qdrantClient.QueryAsync("galaxypedia");


        var pagesToUpdate = new List<(string title, string newcontent, ScoredPoint olddata)>();

        // Should try to do this in linq but I'm going to keep it simple for now
        foreach ((string title, string content) page in pages)
        {
            ScoredPoint? existingPoint = qdrantCollection.FirstOrDefault(p => p.Payload["title"].ToString() == page.title);

            if (existingPoint == null) continue;

            if (existingPoint.Payload["content"].ToString() != page.content) pagesToUpdate.Add((page.title, page.content, existingPoint));
        }

        if (pages.Count == 0) return;

        List<(string title, string content, int tokencount)> chunksList = DatasetCreator.ChunkPages(pages, embeddingsTokenizer);

        List<(string title, string content, int tokencount, float[] embedding)> embeddedChunks = await BatchRequestModule.CreateAndProcessBatchRequest(chunksList, embeddingsModel, openAiClient.GetFileClient(), openAiClient.GetBatchClient());
    }
}