// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

namespace galaxygpt.Database;

public class Metadata
{
    public int Id { get; init; }

    /// <summary>
    ///     The name of the dataset (typically something like "dataset-v1")
    /// </summary>
    public required string DatasetName { get; init; }

    /// <summary>
    ///     The date and time the dataset was created at. Use UTC time.
    /// </summary>
    public required DateTime CreatedAt { get; init; }

    /// <summary>
    ///     The maximum size of each chunk
    /// </summary>
    public required int ChunkMaxSize { get; init; }
}