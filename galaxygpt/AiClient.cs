// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Reflection;
using System.Text.RegularExpressions;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;
using OpenAI.Chat;
using OpenAI.Moderations;

namespace galaxygpt;

public partial class AiClient(
    ChatClient chatClient,
    [FromKeyedServices("gptTokenizer")] TiktokenTokenizer gptTokenizer,
    ContextManager contextManager,
    ModerationClient? moderationClient = null)
{
    // I'll copy the files to build output for now. But in the future, they should probably be embedded into the exe
    private static readonly string OneoffSystemMessage = File.ReadAllText(Path.Combine(
        Path.GetDirectoryName(Assembly.GetEntryAssembly()?.Location) ?? throw new InvalidOperationException(),
        "System Messages", "oneoff.txt"));

    private static readonly string ConversationSystemMessage = File.ReadAllText(
        Path.Combine(
            Path.GetDirectoryName(Assembly.GetEntryAssembly()?.Location) ?? throw new InvalidOperationException(),
            "System Messages", "conversation.txt"));


    /// <summary>
    ///     Answers a question based on the provided context
    /// </summary>
    /// <param name="question">The question to answer</param>
    /// <param name="context">Context to provide the AI to help in answering the question</param>
    /// <param name="maxInputTokens">
    ///     The maximum amount of tokens the question can be before it is refused. If left blank, is
    ///     set to unlimited
    /// </param>
    /// <param name="username">Optionally provide a username to associate the request with</param>
    /// <param name="maxOutputTokens">The maximum amount of tokens to output</param>
    /// <returns>A tuple containing the output and the token count of the output</returns>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    /// <exception cref="BonkedException">The moderation API flagged the response</exception>
    public async Task<(string, int)> AnswerQuestion(string question, string context, int? maxInputTokens = null,
        string? username = null, int? maxOutputTokens = null)
    {
        question = question.Trim();
        CheckQuestion(question, maxInputTokens);
        await ModerateText(question, moderationClient);

        if (!string.IsNullOrWhiteSpace(username))
            username = AlphaNumericRegex().Match(username).Value;

        List<ChatMessage> messages =
        [
            new SystemChatMessage(OneoffSystemMessage),
            new UserChatMessage(
                $"Information:\n{context.Trim()}\n\n---\n\nQuestion: {question}\nUsername: {username ?? "N/A"}")
            {
                ParticipantName = username ?? null
            }
        ];

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(messages,
            new ChatCompletionOptions
            {
                MaxTokens = maxOutputTokens
            });

        messages.Add(new AssistantChatMessage(clientResult));

        string finalMessage = messages[^1].Content[0].Text;
        await ModerateText(finalMessage, moderationClient);
        return (finalMessage, gptTokenizer.CountTokens(finalMessage));
    }

    /// <summary>
    ///     Perform checks on the question to ensure it is valid
    /// </summary>
    /// <remarks>Checks if the question is just whitespace, and optionally if it's too long</remarks>
    /// <param name="question"></param>
    /// <param name="maxInputTokens"></param>
    /// <exception cref="ArgumentException"></exception>
    private void CheckQuestion(string question, int? maxInputTokens)
    {
        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("The question cannot be empty.");

        if (maxInputTokens != null && gptTokenizer.CountTokens(question) > maxInputTokens)
            throw new ArgumentException("The question is too long to be answered.");
    }

    private static async Task ModerateText(string text, ModerationClient? client)
    {
        if (client == null)
        {
            Console.WriteLine(
                "Warning: No moderation client was provided. Skipping moderation check. This can be dangerous");
            return;
        }

        ClientResult<ModerationResult> moderation = await client.ClassifyTextInputAsync(text);

        if (moderation.Value.Flagged)
            throw new BonkedException("The question was flagged by the moderation API");
    }

    /// <summary>
    ///     Continues a conversation. Will pick the last <see cref="UserChatMessage" />, grab the context, then call
    ///     <see cref="AnswerQuestion" />
    /// </summary>
    /// <param name="conversation">The conversation to use</param>
    /// <param name="maxOutputTokens"></param>
    /// <returns>The new <see cref="ChatMessage" /> List</returns>
    public async Task<List<ChatMessage>> FollowUpConversation(List<ChatMessage> conversation,
        int? maxOutputTokens = null)
    {
        // The conversation is unlikely to be in the same format as the one in AnswerQuestion. Notably, there will be no Context or Username. Just the raw question.
        // We can go two ways about this:
        // 1. We can go through and get context for each question. This gives the AI the most information to go off of (it knows what information were used to answer every question)
        // 2. We can just grab the second to last UserChatMessage and grab the context for it, then call AnswerQuestion on the last UserChatMessage. This provides context for the previous question, and the new question. (The AI will know what information was used to answer the previous question)
        // 3. We can grab the last UserChatMessage and call AnswerQuestion on it. This means the AI will only have context for the new question, but it will be more performant. (The AI will have no prior information to go off of except for its own responses)
        // For now, I'll go with option 3 since we can expand to option 2 if needed.

        foreach (ChatMessage message in conversation)
            await ModerateText(message.Content.First().Text, moderationClient);

        UserChatMessage
            lastUserMessage = conversation.OfType<UserChatMessage>().Last(); // this is the new (follow up) question
        string lastQuestion = lastUserMessage.Content.First().Text;

        CheckQuestion(lastQuestion, null);

        // TODO: This is unreliable. We should have the caller specify whether or not the last message contains a context.
        if (!lastQuestion.Contains("context: ", StringComparison.OrdinalIgnoreCase))
        {
            // The last message does not contain a context. We need to find the context.

            string context = (await contextManager.FetchContext(lastQuestion)).Item1;

            conversation.Remove(lastUserMessage);
            conversation.Add(new UserChatMessage($"Question: {lastQuestion}\n\nInformation:\n{context}"));

            // Update lastUserMessage and lastQuestion to point to the new UserChatMessage
            // Wait is this even necessary? We're not using lastUserMessage or lastQuestion after this point.
            // Okay let's comment this out for now and see if it breaks anything.
            // lastUserMessage = conversation.OfType<UserChatMessage>().Last();
            // lastQuestion = lastUserMessage.Content.First().Text;
        }

        conversation.Insert(0, new SystemChatMessage(ConversationSystemMessage));

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(conversation,
            new ChatCompletionOptions
            {
                MaxTokens = maxOutputTokens
            });

        await ModerateText(clientResult.Value.Content[0].Text, moderationClient);

        conversation.Add(new AssistantChatMessage(clientResult));

        // Maybe we shouldn't modify the conversation in place? Don't know if the caller minds that the UserChatMessage
        // was modified or not since all that really matters is the final response from the assistant.
        return conversation;
    }

    [GeneratedRegex("[a-zA-Z0-9_-]+")]
    private static partial Regex AlphaNumericRegex();
}