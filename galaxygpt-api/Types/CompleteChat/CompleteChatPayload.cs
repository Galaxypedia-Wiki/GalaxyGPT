// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ComponentModel;

namespace galaxygpt_api.Types.CompleteChat;

public class CompleteChatPayload
{
    /// <summary>
    /// The messages to use for the request
    /// </summary>
    public required List<ChatMessageGeneric> Conversation { get; init; }

    /// <summary>
    /// The model to use for the request
    /// </summary>
    [DefaultValue("gpt-4o-mini")]
    public string? Model { get; init; } = "gpt-4o-mini";

    /// <summary>
    /// The username of the user asking the question
    /// </summary>
    public string? Username { get; init; }

    /// <summary>
    /// The maximum amount of tokens to generate
    /// </summary>
    public int? MaxLength { get; init; }

    /// <summary>
    /// The maximum amount of pages to pull from the embeddings database
    /// </summary>
    [DefaultValue(5)]
    public ulong? MaxContextLength { get; init; } = 5;

}