// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.Text.Json.Serialization;

namespace galaxygpt_api.Types.CompleteChat;

public class CompleteChatResponse
{
    public required List<ChatMessageGeneric> Conversation { get; init; }
    public required string Version { get; init; }

    /// <summary>
    /// The combined amount of tokens in the system prompt, context, and user's question
    /// </summary>
    [JsonPropertyName("question_tokens")]
    public string? QuestionTokens { get; init; }

    /// <summary>
    /// The amount of tokens in chatgpt's response
    /// </summary>
    [JsonPropertyName("response_tokens")]
    public string? ResponseTokens { get; init; }
}