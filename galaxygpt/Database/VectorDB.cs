// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using Microsoft.EntityFrameworkCore;

namespace galaxygpt.Database;

public class VectorDb(string? path = null) : DbContext
{
    private readonly string _dbPath = "Data Source=" + (path ??
                                                        Path.Join(
                                                            Environment.GetFolderPath(Environment.SpecialFolder
                                                                .LocalApplicationData), "embeddings.db"));

    public DbSet<Page> Pages { get; set; }
    public DbSet<Chunk> Chunks { get; set; }
    public DbSet<Metadata> Metadata { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder options)
    {
        options.UseSqlite(_dbPath);
    }
}