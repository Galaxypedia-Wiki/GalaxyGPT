// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

namespace galaxygpt_api.Types.CompleteChat;

public class ChatMessageGeneric
{
    /// <summary>
    /// Either "user", "assistant", or "system"
    /// </summary>
    public required string Role { get; set; }

    /// <summary>
    /// The message content
    /// </summary>
    public required string Message { get; set; }
}