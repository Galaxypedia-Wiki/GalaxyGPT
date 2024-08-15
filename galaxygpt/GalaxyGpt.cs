// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Data;
using System.Numerics.Tensors;
using System.Text;
using galaxygpt.Database;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.ML.Tokenizers;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using OpenAI.Moderations;

namespace galaxygpt;

public static class GalaxyGpt
{
    // TODO: Migrate to dependency injection
    private static readonly VectorDb Db = new();
    private static readonly IConfigurationRoot Configuration = new ConfigurationBuilder().AddJsonFile("appsettings.json", optional: true, true).AddEnvironmentVariables().Build();
    private static readonly OpenAIClient OpenAiClient = new(Configuration["OPENAI_API_KEY"] ?? throw new InvalidOperationException());
    public static readonly TiktokenTokenizer GptTokenizer = TiktokenTokenizer.CreateForModel("gpt-4o-mini");
    private static readonly TiktokenTokenizer EmbeddingsTokenizer = TiktokenTokenizer.CreateForModel("text-embedding-3-small");

    /// <summary>
    /// Answer a question using the specified model.
    /// </summary>
    /// <param name="question">What questions to ask</param>
    /// <param name="context"></param>
    /// <param name="model">A string of which model to use</param>
    /// <param name="maxInputTokens"></param>
    /// <param name="maxOutputTokens">The maximum amount of tokens to return. ds this number.</param>
    /// <param name="moderationModel"></param>
    /// <param name="username">The username to pass to the bot, used for personalizing the response.</param>
    /// <param name="temperature"></param>
    /// <param name="db">The database to use in the context. This defaults to the GalaxyGPT-wide database, but can be manually set to another as needed.</param>
    public static async Task<string> AnswerQuestion(string question, string context, string model, int maxInputTokens,
        int maxOutputTokens,
        string moderationModel = "text-moderation-stable", string? username = null, int temperature = 1,
        VectorDb? db = null)
    {
        db ??= Db;

        #region Sanitize & Check the question

        question = question.Trim();

        if (string.IsNullOrEmpty(question))
        {
            throw new ArgumentException("The question cannot be empty.");
        }

        if (GptTokenizer.CountTokens(question) > maxInputTokens)
        {
            throw new ArgumentException("The question is too long to be answered.");
        }

        // Check if database is empty
        if (!db.Pages.Any())
        {
            throw new InvalidOperationException("The database is empty. Please load a dataset first.");
        }

        // Throw the question into the moderation API
        ClientResult<ModerationResult>? moderation = await OpenAiClient.GetModerationClient(moderationModel).ClassifyTextInputAsync(question);

        if (moderation.Value.Flagged)
        {
            throw new InvalidOperationException("The question was flagged by the moderation API.");
        }

        #endregion

        ChatClient? chatClient = OpenAiClient.GetChatClient(model);

        List<ChatMessage> messages =
        [
            new SystemChatMessage("""
                                  You are GalaxyGPT, a helpful assistant that answers questions about Galaxy, a ROBLOX Space Game.
                                  The Galaxypedia is the game's official wiki and it is your creator.
                                  The Galaxypedia's slogans are "The new era of the Galaxy Wiki" and "A hub for all things Galaxy".
                                  Answer the question based on the supplied context. If the question cannot be answered, politely say you don't know the answer and ask the user for clarification, or if they have any further questions about Galaxy.
                                  If the user has a username, it will be provided and you can address them by it. If a username is not provided (it shows as N/A), do not address/refer the user apart from "you" or "your".
                                  Do not reference or mention the "context provided" in your response, no matter what.
                                  The context will be given in the format of wikitext. You will be given multiple different pages in your context to work with. The different pages will be separated by "###".
                                  If a ship infobox is present in the context, prefer using data from within the infobox. An infobox can be found by looking for a wikitext template that has the word "infobox" in its name.
                                  If the user is not asking a question (e.g. "thank you", "thanks for the help"): Respond to it and ask the user if they have any further questions.
                                  Respond to greetings (e.g. "hi", "hello") with (in this exact order): A greeting, a brief description of yourself, and a question addressed to the user if they have a question or need assistance.
                                  Above all, be polite and helpful to the user. 
                                  
                                  Steps for responding:
                                  First check if the user is asking about a ship (e.g. "what is the deity?", "how much shield does the theia have?"), if so, use the ship's wiki page (supplied in the context) and the statistics from the ship's infobox to answer the question.
                                  If you determine the user is not asking about a ship (e.g. "who is <player>?", "what is <item>?"), do your best to answer the question based on the context provided.
                                  """),
            new UserChatMessage($"Context:\n{context.Trim()}\n\n---\n\nQuestion: {question}\nUsername: {username ?? "N/A"}")
            {
                ParticipantName = username ?? null
            }
        ];

        ClientResult<ChatCompletion>? idk = await chatClient.CompleteChatAsync(messages, new ChatCompletionOptions
        {
            MaxTokens = maxOutputTokens,
            Temperature = temperature

        });
        messages.Add(new AssistantChatMessage(idk));

        return messages[^1].Content[0].Text;
    }

    /// <summary>
    ///
    /// </summary>
    /// <param name="question"></param>
    /// <param name="model"></param>
    /// <param name="db"></param>
    /// <param name="maxLength">If left null, the top 5 chunks will be returned</param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    public static async Task<(string, int)> FetchContext(string question, string model, VectorDb? db = null,
        int? maxLength = null)
    {
        db ??= Db;
        question = question.Trim();

        if (string.IsNullOrEmpty(question))
            throw new ArgumentException("The question cannot be empty.");

        if (!db.Pages.Any())
            throw new InvalidOperationException("The database is empty. Please load a dataset first.");

        EmbeddingClient? embeddingsClient = OpenAiClient.GetEmbeddingClient(model);
        ClientResult<Embedding>? questionEmbeddings = await embeddingsClient.GenerateEmbeddingAsync(question);

        List<Page> pages = await db.Pages.Include(chunk => chunk.Chunks).ToListAsync();
        var pageEmbeddings = new List<(Page page, float[] embeddings, int chunkId, float distance)>();

        foreach (Page page in pages)
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
        int tokenCount = GptTokenizer.CountTokens(question);
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
                tokenCount += GptTokenizer.CountTokens(content);
                if (tokenCount > maxLength)
                    break;
            }

            context.Append($"Page: {page.Title}\nContent: {content}\n\n###\n\n");
            iterations++;
        }

        return (context.ToString(), EmbeddingsTokenizer.CountTokens(question));
    }
}