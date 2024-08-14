using Microsoft.EntityFrameworkCore;

namespace galaxygpt.Database;

public class VectorDb : DbContext
{
    public DbSet<Page> Pages { get; set; }
    public DbSet<Chunk> Chunks { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
        => options.UseSqlite("Data Source=" + Path.Join(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "embeddings.db"));
}