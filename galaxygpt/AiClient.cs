// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;
using OpenAI.Chat;
using OpenAI.Moderations;

namespace galaxygpt;

public class AiClient(
    ChatClient chatClient,
    [FromKeyedServices("gptTokenizer")] TiktokenTokenizer gptTokenizer,
    ModerationClient? moderationClient = null)
{
    public async Task<string> AnswerQuestion(string question, string context, int maxInputTokens,
        string? username = null, int? maxOutputTokens = null)
    {
        #region Sanitize & Check the question

        question = question.Trim();

        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("The question cannot be empty.");

        if (gptTokenizer.CountTokens(question) > maxInputTokens)
            throw new ArgumentException("The question is too long to be answered.");

        // Throw the question into the moderation API
        if (moderationClient != null)
        {
            ClientResult<ModerationResult> moderation = await moderationClient.ClassifyTextInputAsync(question);

            if (moderation.Value.Flagged)
                throw new InvalidOperationException("The question was flagged by the moderation API.");
        }
        else
        {
            Console.WriteLine(
                "Warning: No moderation client was provided. Skipping moderation check. This can be dangerous");
        }

        #endregion

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
            new UserChatMessage(
                $"Context:\n{context.Trim()}\n\n---\n\nQuestion: {question}\nUsername: {username ?? "N/A"}")
            {
                ParticipantName = username ?? null
            }
        ];

        ClientResult<ChatCompletion>? idk = await chatClient.CompleteChatAsync(messages, new ChatCompletionOptions
        {
            MaxTokens = maxOutputTokens
        });
        messages.Add(new AssistantChatMessage(idk));

        return messages[^1].Content[0].Text;
    }
}