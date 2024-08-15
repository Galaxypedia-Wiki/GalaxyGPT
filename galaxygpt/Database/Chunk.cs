// Copyright (c) smallketchup82. Licensed under the GPL3 Licence.
// See the LICENCE file in the repository root for full licence text.

using Microsoft.EntityFrameworkCore;

namespace galaxygpt.Database;

public class Chunk
{
    public int Id { get; init; }
    public string Content { get; init; }

    public List<float>? Embeddings { get; set; }
}