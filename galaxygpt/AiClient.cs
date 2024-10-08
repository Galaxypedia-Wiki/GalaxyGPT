// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
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
        Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? throw new InvalidOperationException(),
        "System Messages", "oneoff.txt"));

    private static readonly string ConversationSystemMessage = File.ReadAllText(
        Path.Combine(
            Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? throw new InvalidOperationException(),
            "System Messages", "conversation.txt"));


    /// <summary>
    ///     Answers a question based on the provided context
    /// </summary>
    /// <remarks>This and <see cref="FollowUpConversation" /> should probably be merged.</remarks>
    /// <param name="question">The question to answer</param>
    /// <param name="context">Context to provide the AI to help in answering the question</param>
    /// <param name="maxInputTokens">
    ///     The maximum amount of tokens the question can contain before it is refused. If left blank, is
    ///     set to unlimited
    /// </param>
    /// <param name="username">Optionally provide a username to associate the request with</param>
    /// <param name="maxOutputTokens">The maximum amount of tokens to output. This is NOT a hard limit, and can be exceeded.</param>
    /// <returns>A tuple containing the output and the token count of the output</returns>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    /// <exception cref="BonkedException">The moderation API flagged the response</exception>
    public async Task<(string output, int tokencount)> AnswerQuestion(string question, string context,
        int? maxInputTokens = null,
        string? username = null, int? maxOutputTokens = null)
    {
        CheckQuestion(question, maxInputTokens, maxOutputTokens);

        // Start the moderation task
        Task moderateQuestionTask = ModerateText(question, moderationClient);

        if (!string.IsNullOrWhiteSpace(username))
            username = AlphaNumericRegex().Match(username).Value;

        List<ChatMessage> messages =
        [
            new SystemChatMessage(OneoffSystemMessage),
            new UserChatMessage(BuildUserMessage(question, context, username))
            {
                ParticipantName = username != null ? Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(username))) : null
            }
        ];

        // Wait for moderation to finish before continuing
        await moderateQuestionTask;

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(messages,
            new ChatCompletionOptions
            {
                MaxOutputTokenCount = maxOutputTokens
            });

        messages.Add(new AssistantChatMessage(clientResult));

        string finalMessage = messages[^1].Content[0].Text;
        await ModerateText(finalMessage, moderationClient);
        return (finalMessage, gptTokenizer.CountTokens(finalMessage));
    }

    private static string BuildUserMessage(string question, string context, string? username)
    {
        StringBuilder userMessage = new StringBuilder()
            .AppendLine("Information:")
            .AppendLine(context)
            .AppendLine()
            .AppendLine()
            .AppendLine("---")
            .AppendLine()
            .AppendLine()
            .AppendLine($"Question: {question}")
            .AppendLine($"Username: {username ?? "N/A"}");

        return userMessage.ToString();
    }

    /// <summary>
    ///     Perform checks on the question to ensure it is valid
    /// </summary>
    /// <remarks>Checks if the question is just whitespace, and optionally if it's too long</remarks>
    /// <param name="question"></param>
    /// <param name="maxInputTokens"></param>
    /// <param name="maxOutputTokens"></param>
    /// <exception cref="ArgumentException"></exception>
    private void CheckQuestion(string question, int? maxInputTokens, int? maxOutputTokens)
    {
        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("The question cannot be empty");

        if (maxInputTokens != null && gptTokenizer.CountTokens(question) > maxInputTokens)
            throw new ArgumentException("The question is too long to be answered");

        if (maxOutputTokens == 0)
            throw new ArgumentException("The maximum output token count cannot be 0");
    }

    public static async Task ModerateText(string text, ModerationClient? client)
    {
        if (client == null)
        {
            Console.WriteLine(
                "Warning: No moderation client was provided. Skipping moderation check. This can be dangerous");
            return;
        }

        ClientResult<ModerationResult> moderation = await client.ClassifyTextAsync(text);

        if (moderation.Value.Flagged)
            throw new BonkedException("The question was flagged by the moderation API");
    }

    /// <summary>
    ///     Continues a conversation.
    /// </summary>
    /// <param name="conversation">The conversation to use</param>
    /// <param name="context"></param>
    /// <param name="maxOutputTokens"></param>
    /// <returns>The new <see cref="ChatMessage" /> List</returns>
    public async Task<List<ChatMessage>> FollowUpConversation(List<ChatMessage> conversation,
        string? context = null, int? maxOutputTokens = null)
    {
        // The conversation is unlikely to be in the same format as the one in AnswerQuestion. Notably, there will be no Context or Username. Just the raw question.
        // We can go two ways about this:
        // 1. We can go through and get context for each question. This gives the AI the most information to go off of (it knows what information were used to answer every question)
        // 2. We can just grab the second to last UserChatMessage and grab the context for it, then call AnswerQuestion on the last UserChatMessage. This provides context for the previous question, and the new question. (The AI will know what information was used to answer the previous question)
        // 3. We can grab the last UserChatMessage and call AnswerQuestion on it. This means the AI will only have context for the new question, but it will be more performant. (The AI will have no prior information to go off of except for its own responses)
        // For now, I'll go with option 3 since we can expand to option 2 if needed.

        UserChatMessage
            lastUserMessage = conversation.OfType<UserChatMessage>().Last(); // this is the new (follow up) question
        string lastQuestion = lastUserMessage.Content.First().Text;

        // We only need to moderate the last question since it's the only one that is new
        Task moderationTask = ModerateText(lastQuestion, moderationClient);

        CheckQuestion(lastQuestion, null, null);

        // Fetch the context for the last question if it wasn't provided
        context ??= (await contextManager.FetchContext(lastQuestion)).Item1;

        // Remove the last UserChatMessage and add a new one with the context
        conversation.Remove(lastUserMessage);
        conversation.Add(new UserChatMessage($"Question: {lastQuestion}\n\nInformation:\n{context}"));

        conversation.Insert(0, new SystemChatMessage(ConversationSystemMessage));

        await moderationTask;

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(conversation,
            new ChatCompletionOptions
            {
                MaxOutputTokenCount = maxOutputTokens
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