// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using OpenAI.Batch;
using OpenAI.Files;
using Spectre.Console;

namespace dataset_assistant;

/// <summary>
/// High level class that manages sending, receiving, and parsing batch requests
/// </summary>
public static class BatchRequestModule
{
    /// <summary>
    /// Creates, makes, processes, and formats a batch request, returning a list of the embedded chunks with their titles, content, token count, and embedding
    /// </summary>
    /// <param name="chunksList">A tuple containing pages/chunks and their contents (tokencount is required but not used)</param>
    /// <param name="embeddingsModel">The model to use</param>
    /// <param name="fileClient">The <see cref="FileClient"/> to use</param>
    /// <param name="batchClient">The <see cref="BatchClient"/> to use</param>
    /// <param name="progressTask">A <see cref="ProgressTask"/> for use in notifying progress. Optional</param>
    /// <returns></returns>
    public static async Task<List<(string title, string content, int tokencount, float[] embedding)>> CreateAndProcessBatchRequest(List<(string title, string content, int tokencount)> chunksList,
        string embeddingsModel, FileClient fileClient, BatchClient batchClient, ProgressTask? progressTask = null)
    {
        string batchRequestsString = await CreateBatchRequest(chunksList, embeddingsModel);

        string? batchId = await MakeBatchRequest(fileClient, batchClient, batchRequestsString);

        // Wait 3 seconds before checking the batch status to give the api some time to process the request
        await Task.Delay(3000);

        string? batchOutputFileId = await WaitForBatchRequestCompletion(batchClient, progressTask, batchId);

        ClientResult<BinaryData>? batchOutputFile = await fileClient.DownloadFileAsync(batchOutputFileId!);

        await using var fileStream = batchOutputFile.Value.ToStream();

        List<(string title, string content, int tokencount, float[] embedding)> embeddedChunksList = await FormatBatchResults(chunksList, fileStream);

        progressTask?.Increment(1);

        return embeddedChunksList;
    }

    private static async Task<List<(string title, string content, int tokencount, float[] embedding)>> FormatBatchResults(List<(string title, string content, int tokencount)> chunksList, Stream fileStream)
    {
        var responses = new List<string>();

        using var streamReader = new StreamReader(fileStream);
        while (!streamReader.EndOfStream)
        {
            string? line = await streamReader.ReadLineAsync();
            if (line != null)
                responses.Add(line);
        }

        // Now we have the responses, we should convert each to a jsonelement
        List<JsonElement> responsesJson = responses.Select(response => JsonSerializer.Deserialize<JsonElement>(response)).ToList();

        // Now we have the json elements, we can extract the embeddings from them
        // We don't care about anything else in the response except for the title and the embedding, so let's use a dictionary
        var embeddedChunks = new Dictionary<string, float[]>();

        foreach (JsonElement response in responsesJson)
        {
            string? title = response.GetProperty("custom_id").GetString();
            float[] embedding = response.GetProperty("response")
                .GetProperty("body")
                .GetProperty("data")
                .EnumerateArray()
                .First()
                .GetProperty("embedding")
                .EnumerateArray()
                .Select(e => e.GetSingle())
                .ToArray();
            embeddedChunks.Add(title ?? throw new InvalidOperationException(), embedding);
        }
        
        // Now we have the embedded chunks, lets create a new list, taking the original chunks and adding the embedding to it
        List<(string title, string content, int tokencount, float[] embedding)> embeddedChunksList = chunksList.Select(chunk =>
        {
            string title = chunk.title;
            string content = chunk.content;
            int tokencount = chunk.tokencount;
            float[] embedding = embeddedChunks[title];
            return (title, content, tokencount, embedding);
        }).ToList();
        return embeddedChunksList;
    }

    private static async Task<string?> WaitForBatchRequestCompletion(BatchClient batchClient, ProgressTask? progressTask,
        string? batchId)
    {
        string? batchOutputFileId = null;

        while (true)
        {
            var batchJson = (await batchClient.GetBatchAsync(batchId, null)).GetRawResponse()
                .Content.ToObjectFromJson<JsonElement>();

            // WARNING: If any of these properties are null, the program will crash
            // We should replace them with TryGetProperty and handle the null case, or wrap this in a try-catch block
            string? batchStatus = batchJson.GetProperty("status").GetString();
            int batchTotalTasks = batchJson.GetProperty("request_counts").GetProperty("total").GetInt32();
            int batchCompletedTasks = batchJson.GetProperty("request_counts").GetProperty("completed").GetInt32();
            batchOutputFileId ??= batchJson.GetProperty("output_file_id").GetString();

            if (batchStatus == "completed")
                break;

            // Set the max value to the total number of tasks + 1 to account for the final task of downloading the output file
            if (progressTask != null)
            {
                progressTask.MaxValue(batchTotalTasks+1);
                progressTask.Value(batchCompletedTasks);
            }

            // Poll in 10 seconds
            await Task.Delay(10000);
        }

        return batchOutputFileId;
    }

    private static async Task<string?> MakeBatchRequest(FileClient fileClient, BatchClient batchClient, string batchRequestsString)
    {
        await using var batchRequestsStream = new MemoryStream(Encoding.UTF8.GetBytes(batchRequestsString));

        ClientResult<OpenAIFileInfo>? file =
            await fileClient.UploadFileAsync(batchRequestsStream, "batch_requests.json", FileUploadPurpose.Batch);

        BinaryData input = BinaryData.FromBytes(Encoding.UTF8
            .GetBytes(
                $" {{ \"input_file_id\": \"{file.Value.Id}\", \"endpoint\": \"/v1/embeddings\", \"completion_window\": \"24h\" }} ")
            .ToArray());

        ClientResult? batch = await batchClient.CreateBatchAsync(BinaryContent.Create(input));

        string? batchId = batch.GetRawResponse().Content.ToObjectFromJson<JsonElement>().GetProperty("id")
            .GetString();
        return batchId;
    }

    private static async Task<string> CreateBatchRequest(List<(string title, string content, int tokencount)> chunksList, string embeddingsModelOptionValue)
    {
        var batchRequests = new ConcurrentBag<string>();

        // We can do this in parallel because the requests are independent of each other, it doesn't matter what the order is
        await Parallel.ForEachAsync(chunksList, async (chunk, token) =>
        {
            var batchRequest = new BatchRequest
            {
                CustomId = chunk.title,
                Method = "POST",
                Url = "/v1/embeddings",
                Body =
                    new Dictionary<string, string>
                    {
                        { "input", chunk.content },
                        { "model", embeddingsModelOptionValue }
                    }
            };
            await using var memoryStream = new MemoryStream();
            await JsonSerializer.SerializeAsync(memoryStream, batchRequest, cancellationToken: token);
            string value = Encoding.UTF8.GetString(memoryStream.ToArray());
            batchRequests.Add(value);
        });

        string batchRequestsString = string.Join("\n", batchRequests);

        if (string.IsNullOrEmpty(batchRequestsString)) throw new InvalidOperationException("Batch requests string is empty");
        return batchRequestsString;
    }
}