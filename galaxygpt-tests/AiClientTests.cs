// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using System.ClientModel.Primitives;
using System.Text.Json;
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
    private static readonly Mock<ChatClient> ChatClientMock = new();

    private static readonly Mock<ModerationClient> ModerationClientMock = new();

    private static readonly Mock<EmbeddingClient> EmbeddingClientMock = new();

    private static readonly Mock<ContextManager> ContextManagerMock = new(EmbeddingClientMock.Object,
        TiktokenTokenizer.CreateForModel("text-embedding-3-small"), null!);

    private static ITestOutputHelper _output = null!;

    private readonly AiClient _aiClient;

    public AiClientTests(ITestOutputHelper output)
    {
        ChatCompletion chatCompletion = OpenAIChatModelFactory.ChatCompletion(content:
        [
            ChatMessageContentPart.CreateTextPart("goofy ahh uncle productions")
        ], role: ChatMessageRole.Assistant);

        Mock<ClientResult<ChatCompletion>> chatClientResultMock = new(null!, Mock.Of<PipelineResponse>());

        chatClientResultMock
            .SetupGet(result => result.Value)
            .Returns(chatCompletion);

        ChatClientMock.Setup(client => client.CompleteChatAsync(
            It.IsAny<List<ChatMessage>>(),
            It.IsAny<ChatCompletionOptions>(),
            It.IsAny<CancellationToken>()
        )).Returns(Task.FromResult(chatClientResultMock.Object));

        ModerationResult moderationResult = OpenAIModerationsModelFactory.ModerationResult();

        Mock<ClientResult<ModerationResult>> moderationClientResultMock = new(null!, Mock.Of<PipelineResponse>());

        moderationClientResultMock
            .SetupGet(result => result.Value)
            .Returns(moderationResult);

        ModerationClientMock
            .Setup(client => client.ClassifyTextAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .Returns(Task.FromResult(moderationClientResultMock.Object));

        _aiClient = new AiClient(ChatClientMock.Object, TiktokenTokenizer.CreateForModel("text-embedding-3-small"),
            ContextManagerMock.Object, ModerationClientMock.Object);

        _output = output;
    }

    [Fact]
    public async Task TestAnswersQuestion()
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
    public async Task TestAnswersQuestionWithZeroMaxOutputTokens()
    {
        // Arrange
        const string question = "What is the meaning of life?";
        const string context =
            "The meaning of life is a philosophical question concerning the significance of life or existence in general.";
        int? maxInputTokens = 100;
        const string username = "smallketchup82";
        int? maxOutputTokens = 0;

        // Act
        await Assert.ThrowsAsync<ArgumentException>(() => _aiClient.AnswerQuestion(question, context, maxInputTokens, username, maxOutputTokens));
    }

    [Fact]
    public async Task TestModeratesText()
    {
        ModerationClientMock.Invocations.Clear();
        // Arrange
        const string text = "goofy ahh uncle productions";

        // Act
        await AiClient.ModerateText(text, ModerationClientMock.Object);

        // Assert
        ModerationClientMock.Verify(client => client.ClassifyTextAsync(text, It.IsAny<CancellationToken>()),
            Times.Once);
    }

    [Fact]
    public async Task TestModeratesTextWithoutClient()
    {
        ModerationClientMock.Invocations.Clear();

        // Arrange
        const string text = "goofy ahh uncle productions";

        // Act
        await AiClient.ModerateText(text, null);

        // Assert
        ModerationClientMock.Verify(client => client.ClassifyTextAsync(text, It.IsAny<CancellationToken>()),
            Times.Never);
    }

    [Fact]
    public async Task TestModeratedText()
    {
        // We need to set up a custom moderation result & client result for this test since we need to set the flagged property to true
        ModerationResult? moderationResult = OpenAIModerationsModelFactory.ModerationResult(true);
        var moderationClientResult = new Mock<ClientResult<ModerationResult>>(null!, Mock.Of<PipelineResponse>());
        var moderationClientMock = new Mock<ModerationClient>();

        moderationClientResult
            .SetupGet(result => result.Value)
            .Returns(moderationResult);

        moderationClientMock
            .Setup(client => client.ClassifyTextAsync(It.IsAny<string>(), It.IsAny<CancellationToken>()))
            .ReturnsAsync(moderationClientResult.Object);

        const string text = "goofy ahh uncle productions";

        await Assert.ThrowsAsync<BonkedException>(() => AiClient.ModerateText(text, moderationClientMock.Object));
    }

    [Fact]
    public async Task CheckQuestionThrowsArgumentExceptionWhenQuestionIsWhitespace()
    {
        // Arrange
        const string question = " ";

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _aiClient.AnswerQuestion(question, "context"));
    }

    [Fact]
    public async Task CheckQuestionThrowsArgumentExceptionWhenQuestionIsTooLong()
    {
        // Arrange
        const string question = "What is the meaning of life?";
        const string context =
            "The meaning of life is a philosophical question concerning the significance of life or existence in general.";
        int? maxInputTokens = 1;

        // Act & Assert
        await Assert.ThrowsAsync<ArgumentException>(() => _aiClient.AnswerQuestion(question, context, maxInputTokens));
    }

    [Fact]
    public async Task FollowUpConversationTest()
    {
        // Okay this is a *little* confusing. The way this function works is by taking in a list of ChatMessages and adding a new AssistantChatMessage to the end of it.
        // Because of that, using the same client & mocking setup for AnswerQuestion is fine here. Both should return the same thing in the end
        List<ChatMessage> conversation =
        [
            new UserChatMessage("goofy ahh uncle productions."),
            new AssistantChatMessage("What the sigma?"),
            // Use Context: to disable fetching context for now, as it is broken
            new UserChatMessage("try not to say N word challenge. Context: ")
        ];

        List<ChatMessage> test = await _aiClient.FollowUpConversation(conversation);

        Assert.NotNull(test);
        Assert.True(test.OfType<SystemChatMessage>().Count() == 1);
        Assert.True(test.OfType<UserChatMessage>().Any());
        Assert.True(test.OfType<AssistantChatMessage>().Any());

        _output.WriteLine(JsonSerializer.Serialize(test, new JsonSerializerOptions { WriteIndented = true }));
    }
}