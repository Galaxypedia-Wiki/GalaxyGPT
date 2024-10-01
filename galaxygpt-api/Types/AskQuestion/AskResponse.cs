// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.Text.Json.Serialization;

namespace galaxygpt_api.Types.AskQuestion;

public class AskResponse
{
    public required string Answer { get; init; }
    public required string Context { get; init; }
    public required string Duration { get; init; }
    public required string Version { get; init; }
    
    /// <summary>
    /// The amount of tokens in the system prompt
    /// </summary>
    [JsonPropertyName("context_tokens")]
    public required string PromptTokens { get; init; }

    /// <summary>
    /// The amount of tokens in the context
    /// </summary>
    [JsonPropertyName("context_tokens")]
    public required string ContextTokens { get; init; }

    /// <summary>
    /// The amount of tokens in the user's question
    /// </summary>
    [JsonPropertyName("question_tokens")]
    public required string QuestionTokens { get; init; }
    
    /// <summary>
    /// The amount of tokens in chatgpt's response
    /// </summary>
    [JsonPropertyName("response_tokens")]
    public required string ResponseTokens { get; init; }
}