// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.ClientModel.Primitives;
using galaxygpt;
using Microsoft.ML.Tokenizers;
using Moq;
using OpenAI.Chat;
using OpenAI.Embeddings;
using OpenAI.Moderations;
using Xunit.Abstractions;

namespace galaxygpt_tests;

public class AiClientTests
{
    private static Mock<ChatClient> _chatClientMock = new();
    private static Mock<ClientResult<ChatCompletion>> _chatClientResultMock = new(null!, Mock.Of<PipelineResponse>());

    private static ChatCompletion _chatCompletion = OpenAIChatModelFactory.ChatCompletion(content:
    [
        ChatMessageContentPart.CreateTextPart("goofy ahh uncle productions")
    ], role: ChatMessageRole.Assistant);

    private static Mock<ModerationClient> _moderationClientMock = new();

    private static Mock<ClientResult<ModerationResult>> _moderationClientResultMock =
        new(null!, Mock.Of<PipelineResponse>());

    private static ModerationResult _moderationResult = OpenAIModerationsModelFactory.ModerationResult();

    private static Mock<EmbeddingClient> _embeddingClientMock = new();

    private static Mock<ContextManager> _contextManagerMock = new(_embeddingClientMock.Object,
        TiktokenTokenizer.CreateForModel("text-embedding-3-small"), null!);

    private AiClient _aiClient;

    private static ITestOutputHelper _output = null!;

    public AiClientTests(ITestOutputHelper output)
    {
        _chatClientResultMock
            .SetupGet(result => result.Value)
            .Returns(_chatCompletion);

        _chatClientMock.Setup(client => client.CompleteChatAsync(
            It.IsAny<List<ChatMessage>>(),
            It.IsAny<ChatCompletionOptions>(),
            It.IsAny<CancellationToken>()
        )).Returns(Task.FromResult(_chatClientResultMock.Object));

        _moderationClientResultMock
            .SetupGet(result => result.Value)
            .Returns(_moderationResult);

        _moderationClientMock
            .Setup(client => client.ClassifyTextAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .Returns(Task.FromResult(_moderationClientResultMock.Object));

        _aiClient = new AiClient(_chatClientMock.Object, TiktokenTokenizer.CreateForModel("text-embedding-3-small"),
            _contextManagerMock.Object, _moderationClientMock.Object);

        _output = output;
    }

    [Fact]
    public async void TestAnswersQuestion()
    {
        // Arrange
        const string question = "What is the meaning of life?";
        const string context =
            "The meaning of life is a philosophical question concerning the significance of life or existence in general.";
        int? maxInputTokens = 100;
        const string username = "smallketchup82";
        int? maxOutputTokens = 100;

        // Act
        (string output, int tokencount) result =
            await _aiClient.AnswerQuestion(question, context, maxInputTokens, username, maxOutputTokens);

        // Assert
        Assert.NotNull(result.output);
        Assert.False(string.IsNullOrWhiteSpace(result.output));
        Assert.True(result.tokencount > 0);
        _output.WriteLine(result.Item1);
    }

    [Fact]
    public async void TestModeratesText()
    {
        _moderationClientMock.Invocations.Clear();
        // Arrange
        const string text = "goofy ahh uncle productions";

        // Act
        await AiClient.ModerateText(text, _moderationClientMock.Object);

        // Assert
        _moderationClientMock.Verify(client => client.ClassifyTextAsync(text, It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async void TestModeratesTextWithoutClient()
    {
        _moderationClientMock.Invocations.Clear();

        // Arrange
        const string text = "goofy ahh uncle productions";

        // Act
        await AiClient.ModerateText(text, null);

        // Assert
        _moderationClientMock.Verify(client => client.ClassifyTextAsync(text, It.IsAny<CancellationToken>()),
            Times.Never);
    }

    [Fact]
    public async void TestModeratedText()
    {
        // We need to set up a custom moderation result & client result for this test since we need to set the flagged property to true
        ModerationResult? moderationResult = OpenAIModerationsModelFactory.ModerationResult(flagged: true);
        var moderationClientResult = new Mock<ClientResult<ModerationResult>>(null!, Mock.Of<PipelineResponse>());
        var moderationClientMock = new Mock<ModerationClient>();

        moderationClientResult
            .SetupGet(result => result.Value)
            .Returns(moderationResult);

        moderationClientMock
            .Setup(client => client.ClassifyTextAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .Returns(Task.FromResult(moderationClientResult.Object));

        const string text = "goofy ahh uncle productions";

        await Assert.ThrowsAsync<BonkedException>(() => AiClient.ModerateText(text, moderationClientMock.Object));
    }

    [Fact]
    public async void CheckQuestionThrowsArgumentExceptionWhenQuestionIsWhitespace()
    {
        // Arrange
        const string question = " ";

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _aiClient.AnswerQuestion(question, "context"));
    }

    [Fact]
    public async void CheckQuestionThrowsArgumentExceptionWhenQuestionIsTooLong()
    {
        // Arrange
        const string question = "What is the meaning of life?";
        const string context =
            "The meaning of life is a philosophical question concerning the significance of life or existence in general.";
        int? maxInputTokens = 1;

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _aiClient.AnswerQuestion(question, context, maxInputTokens));
    }
}