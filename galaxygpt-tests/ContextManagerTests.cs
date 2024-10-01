/*
// Copyright (c) smallketchup82.Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.ClientModel.Primitives;
using galaxygpt;
using Microsoft.ML.Tokenizers;
using Moq;
using OpenAI.Embeddings;
using Qdrant.Client;
using Qdrant.Client.Grpc;

namespace galaxygpt_tests
{
    public class ContextManagerTests
    {
        [Fact]
        public async Task FetchContext_ReturnsExpectedContextAndTokenCount()
        {
            var embeddingClientMock = new Mock<EmbeddingClient>();
            var qdrantClientMock = new Mock<QdrantClient>("localhost", 6334);
            var embeddingClientResult = new Mock<ClientResult<Embedding>>(null!, Mock.Of<PipelineResponse>());

            embeddingClientResult.SetupGet(result => result.Value).Returns(OpenAIEmbeddingsModelFactory.Embedding());

            embeddingClientMock
                .Setup(client => client.GenerateEmbeddingAsync(It.IsAny<string>(),
                    It.IsAny<EmbeddingGenerationOptions>(), It.IsAny<CancellationToken>()))
                .ReturnsAsync(embeddingClientResult.Object);

            qdrantClientMock
                .Setup(client => client.QueryAsync(
                    It.IsAny<string>(),
                    It.IsAny<float[]>(),
                    It.IsAny<IReadOnlyList<PrefetchQuery>>(),
                    It.IsAny<string>(),
                    It.IsAny<Filter>(),
                    It.IsAny<float>(),
                    It.IsAny<SearchParams>(),
                    It.IsAny<ulong>(),
                    It.IsAny<ulong>(),
                    It.IsAny<WithPayloadSelector>(),
                    It.IsAny<WithVectorsSelector>(),
                    It.IsAny<ReadConsistency>(),
                    It.IsAny<ShardKeySelector>(),
                    It.IsAny<LookupLocation>(),
                    It.IsAny<TimeSpan>(),
                    It.IsAny<CancellationToken>()
                ))
                .ReturnsAsync(new List<ScoredPoint>
                {
                    new ScoredPoint
                    {
                        Payload =
                        {
                            { "title", "Test Title" },
                            { "content", "Test Content" }
                        },
                        Id = Guid.NewGuid(),
                        Vectors = new Vectors()
                        {
                            Vector = new Vector()
                            {
                                Data =
                                {
                                    0.1f, 0.2f, 0.3f
                                }
                            }
                        }
                    }
                });

            var contextManager = new ContextManager(embeddingClientMock.Object, TiktokenTokenizer.CreateForModel("text-embedding-3-small"), "localhost:6334");

            // Act
            var (context, tokenCount) = await contextManager.FetchContext("Test question");

            // Assert
            Assert.Equal("Page: Test Title\nContent: Test Content\n\n###\n\n", context);
            Assert.Equal(3, tokenCount);
        }

        [Fact]
        public async Task FetchContext_ThrowsInvalidOperationException_WhenQdrantClientIsNull()
        {
            // Arrange
            var embeddingClientMock = new Mock<EmbeddingClient>();
            var embeddingsTokenizerMock = new Mock<TiktokenTokenizer>();

            var contextManager = new ContextManager(embeddingClientMock.Object, embeddingsTokenizerMock.Object, null);

            // Act & Assert
            await Assert.ThrowsAsync<InvalidOperationException>(() => contextManager.FetchContext("Test question"));
        }

        [Fact]
        public async Task FetchContext_ThrowsArgumentException_WhenQuestionIsEmpty()
        {
            // Arrange
            var embeddingClientMock = new Mock<EmbeddingClient>();
            var embeddingsTokenizerMock = new Mock<TiktokenTokenizer>();
            var qdrantClientMock = new Mock<QdrantClient>("localhost", 6334);

            var contextManager = new ContextManager(embeddingClientMock.Object, embeddingsTokenizerMock.Object, "localhost:6334");

            // Act & Assert
            await Assert.ThrowsAsync<ArgumentException>(() => contextManager.FetchContext(" "));
        }
    }
}
*/
