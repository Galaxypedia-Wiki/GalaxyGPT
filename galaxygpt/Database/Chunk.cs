// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

namespace galaxygpt.Database;

public class Chunk
{
    public int Id { get; init; }
    public string Content { get; init; }

    public List<float>? Embeddings { get; set; }
}