// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.Text.Json.Serialization;

namespace dataset_assistant;

public class BatchRequest
{
    [JsonPropertyName("custom_id")]
    public required string CustomId { get; set; }

    [JsonPropertyName("method")]
    public required string Method { get; set; }

    [JsonPropertyName("url")]
    public required string Url { get; set; }

    [JsonPropertyName("body")]
    public required Dictionary<string, string> Body { get; set; }
}