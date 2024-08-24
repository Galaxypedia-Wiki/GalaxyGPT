// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Numerics.Tensors;
using System.Text;
using galaxygpt.Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Embeddings;

namespace galaxygpt;

/// <summary>
/// Handles context management
/// </summary>
public class ContextManager(VectorDb db, EmbeddingClient embeddingClient, [FromKeyedServices("gptTokenizer")] TiktokenTokenizer gptTokenizer, [FromKeyedServices("embeddingsTokenizer")] TiktokenTokenizer embeddingsTokenizer)
{
    /// <summary>
    /// Load all pages from the database into memory
    /// </summary>
    /// <remarks>
    /// Honestly, I tried to avoid this, but considering we'll be doing cosine similarity on everything anyway, it's better to load everything into memory.
    /// </remarks>
    private List<Page> _pages = db.Pages.Include(chunk => chunk.Chunks).ToList();

    public async Task<(string, int)> FetchContext(string question, int? maxLength = null)
    {
        question = question.Trim();

        if (string.IsNullOrEmpty(question))
            throw new ArgumentException("The question cannot be empty.");

        if (!db.Pages.Any())
            throw new InvalidOperationException("The database is empty. Please load a dataset first.");

        ClientResult<Embedding>? questionEmbeddings = await embeddingClient.GenerateEmbeddingAsync(question);

        var pageEmbeddings = new List<(Page page, float[] embeddings, int chunkId, float distance)>();

        foreach (Page page in db.Pages.Include(chunk => chunk.Chunks))
        {
            if (page.Chunks == null || page.Chunks.Count == 0)
            {
                if (page.Embeddings == null) continue;

                float distance = TensorPrimitives.CosineSimilarity(questionEmbeddings.Value.Vector.ToArray(), page.Embeddings.ToArray());
                pageEmbeddings.Add((page, page.Embeddings.ToArray(), -1, distance));
            }
            else if (page.Chunks != null)
            {
                foreach (Chunk chunk in page.Chunks)
                {
                    if (chunk.Embeddings == null) continue;

                    float distance = TensorPrimitives.CosineSimilarity(questionEmbeddings.Value.Vector.ToArray(), chunk.Embeddings.ToArray());
                    pageEmbeddings.Add((page, chunk.Embeddings.ToArray(), chunk.Id, distance));
                }
            }
        }

        pageEmbeddings.Sort((a, b) => b.distance.CompareTo(a.distance));

        StringBuilder context = new();
        int tokenCount = gptTokenizer.CountTokens(question);
        int iterations = 0;

        foreach ((Page page, float[] _, int chunkId, float _) in pageEmbeddings)
        {
            string content = chunkId == -1|| page.Chunks == null || page.Chunks.Count == 0 ? page.Content : page.Chunks.First(chunk => chunk.Id == chunkId).Content;

            if (maxLength == null)
            {
                if (iterations >= 5)
                    break;
            }
            else
            {
                tokenCount += gptTokenizer.CountTokens(content);
                if (tokenCount > maxLength)
                    break;
            }

            context.Append($"Page: {page.Title}\nContent: {content}\n\n###\n\n");
            iterations++;
        }

        return (context.ToString(), embeddingsTokenizer.CountTokens(question));
    }
}