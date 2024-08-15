using Microsoft.EntityFrameworkCore;

namespace galaxygpt.Database;

public class VectorDb(string? path = null) : DbContext
{
    public DbSet<Page> Pages { get; set; }
    public DbSet<Chunk> Chunks { get; set; }
    private readonly string _dbPath = "Data Source=" + (path ?? Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "embeddings.db"));

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite(_dbPath);
}