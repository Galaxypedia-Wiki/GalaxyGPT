// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ComponentModel.DataAnnotations;
using Microsoft.ML.Tokenizers;

namespace galaxygpt.Database;

public class Page
{
    public int Id { get; init; }

    [Required] public required string Title { get; init; }

    // In the future, we might want to store the entire content as a chunk to simplify logic.
    [Required] public required string Content { get; init; }

    // TODO: Remove the nullable and instead initialize it as an empty list.
    public List<Chunk>? Chunks { get; set; }

    public int Tokens { get; init; }

    public List<float>? Embeddings { get; set; }
}