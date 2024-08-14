using Microsoft.EntityFrameworkCore;

namespace galaxygpt.Database;

public class Chunk
{
    public int Id { get; init; }
    public string Content { get; init; }

    public List<float>? Embeddings { get; set; }
}