// Copyright (c) smallketchup82. Licensed under the GPLv3 Licence.
// See the LICENCE file in the repository root for full licence text.

using System.ClientModel;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML.Tokenizers;
using OpenAI.Chat;
using OpenAI.Moderations;

namespace galaxygpt;

public class AiClient(
    ChatClient chatClient,
    [FromKeyedServices("gptTokenizer")] TiktokenTokenizer gptTokenizer,
    ContextManager contextManager,
    ModerationClient? moderationClient = null)
{
    /// <summary>
    /// Answers a question based on the provided context.
    /// </summary>
    /// <param name="question"></param>
    /// <param name="context"></param>
    /// <param name="maxInputTokens"></param>
    /// <param name="username"></param>
    /// <param name="maxOutputTokens"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    /// <exception cref="InvalidOperationException"></exception>
    public async Task<(string, int)> AnswerQuestion(string question, string context, int? maxInputTokens = null,
        string? username = null, int? maxOutputTokens = null)
    {
        question = question.Trim();
        await SanitizeQuestion(question, maxInputTokens);

        List<ChatMessage> messages =
        [
            new SystemChatMessage("""
                                  You are GalaxyGPT, a helpful assistant that answers questions about Galaxy, a ROBLOX Space Game.
                                  The Galaxypedia is the game's official wiki and it is your creator.
                                  The Galaxypedia's slogans are "The new era of the Galaxy Wiki" and "A hub for all things Galaxy".
                                  Answer the question based on the supplied information. If the question cannot be answered, politely say you don't know the answer and ask the user for clarification, or if they have any further questions about Galaxy.
                                  If the user has a username, it will be provided and you can address them by it. If a username is not provided (it shows as N/A), do not address/refer the user apart from "you" or "your".
                                  Do not reference or mention the "information provided" in your response, no matter what.
                                  The information will be given in the format of wikitext. You will be given multiple different pages in your information to work with. The different pages will be separated by "###".
                                  If a ship infobox is present in the information, prefer using data from within the infobox. An infobox can be found by looking for a wikitext template that has the word "infobox" in its name.
                                  If the user is not asking a question (e.g. "thank you", "thanks for the help"): Respond to it and ask the user if they have any further questions.
                                  Respond to greetings (e.g. "hi", "hello") with (in this exact order): A greeting, a brief description of yourself, and a question addressed to the user if they have a question or need assistance.
                                  Above all, be polite and helpful to the user. 

                                  Steps for responding:
                                  First check if the user is asking about a ship (e.g. "what is the deity?", "how much shield does the theia have?"), if so, use the ship's wiki page (supplied in the information) and the statistics from the ship's infobox to answer the question.
                                  If you determine the user is not asking about a ship (e.g. "who is <player>?", "what is <item>?"), do your best to answer the question based on the information provided.
                                  """),
            new UserChatMessage(
                $"Information:\n{context.Trim()}\n\n---\n\nQuestion: {question}\nUsername: {username ?? "N/A"}")
            {
                ParticipantName = username ?? null
            }
        ];

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(messages, new ChatCompletionOptions
        {
            MaxTokens = maxOutputTokens
        });
        messages.Add(new AssistantChatMessage(clientResult));

        string finalMessage = messages[^1].Content[0].Text;
        return (finalMessage, gptTokenizer.CountTokens(finalMessage));
    }

    private async Task SanitizeQuestion(string question, int? maxInputTokens)
    {
        if (string.IsNullOrWhiteSpace(question))
            throw new ArgumentException("The question cannot be empty.");

        if (maxInputTokens != null && gptTokenizer.CountTokens(question) > maxInputTokens)
            throw new ArgumentException("The question is too long to be answered.");

        // Throw the question into the moderation API
        if (moderationClient != null)
        {
            ClientResult<ModerationResult> moderation = await moderationClient.ClassifyTextInputAsync(question);

            if (moderation.Value.Flagged)
                throw new InvalidOperationException("The question was flagged by the moderation API.");
        }
        else
        {
            Console.WriteLine(
                "Warning: No moderation client was provided. Skipping moderation check. This can be dangerous");
        }
    }

    /// <summary>
    /// Continues a conversation. Will pick the last <see cref="UserChatMessage"/>, grab the context, then call <see cref="AnswerQuestion"/>
    /// </summary>
    /// <param name="conversation"></param>
    /// <returns>The new <see cref="ChatMessage"/> List</returns>
    public async Task<List<ChatMessage>> FollowUpConversation(List<ChatMessage> conversation)
    {
        // The conversation is unlikely to be in the same format as the one in AnswerQuestion. Notably, there will be no Context or Username. Just the raw question.
        // We can go two ways about this:
        // 1. We can go through and get context for each question. This gives the AI the most information to go off of (it knows what information were used to answer every question)
        // 2. We can just grab the second to last UserChatMessage and grab the context for it, then call AnswerQuestion on the last UserChatMessage. This provides context for the previous question, and the new question. (The AI will know what information was used to answer the previous question)
        // 3. We can grab the last UserChatMessage and call AnswerQuestion on it. This means the AI will only have context for the new question, but it will be more performant. (The AI will have no prior information to go off of except for its own responses)
        // For now, I'll go with option 3 since we can expand to option 2 if needed.

        UserChatMessage lastUserMessage = conversation.OfType<UserChatMessage>().Last(); // this is the new (follow up) question
        string lastQuestion = lastUserMessage.Content.First().Text;

        await SanitizeQuestion(lastQuestion, null);

        // TODO: This is unreliable. We should have the caller specify whether or not the last message contains a context.
        if (!lastQuestion.Contains("Context: "))
        {
            // The last message does not contain a context. We need to find the context.

            string context = (await contextManager.FetchContext(lastQuestion)).Item1;

            conversation.Remove(lastUserMessage);
            conversation.Add(new UserChatMessage($"Question: {lastQuestion}\n\nInformation:\n{context}"));

            // Update lastUserMessage and lastQuestion to point to the new UserChatMessage
            lastUserMessage = conversation.OfType<UserChatMessage>().Last();
            lastQuestion = lastUserMessage.Content.First().Text;
        }

        conversation.Insert(0, new SystemChatMessage("""
                                                     You are GalaxyGPT, a helpful assistant that answers questions about Galaxy, a ROBLOX Space Game.
                                                     The Galaxypedia is the game's official wiki and it is your creator.
                                                     The Galaxypedia's slogans are "The new era of the Galaxy Wiki" and "A hub for all things Galaxy".

                                                     You have been given a conversation between you and a user. You have already given a response, but the user has asked a follow up question.
                                                     Answer the followup question based on information provided in the conversation. If the question cannot be answered, politely say you don't know the answer and ask the user for clarification, or if they have any other questions about Galaxy.
                                                     You will be given a information to assist in answering the question, but information from the conversation should be preferred. The information should only be used to assist in answering the question, not as the primary source of information.


                                                     If the user has a username, it will be provided and you can address them by it. If a username is not provided (it shows as N/A), do not address/refer the user apart from "you" or "your".
                                                     Do not reference or mention the "information provided" in your response, no matter what.
                                                     The information will be given in the format of wikitext. You will be given multiple different pages in your information to work with. The different pages will be separated by "###".
                                                     If a ship infobox is present in the information, prefer using data from within the infobox. An infobox can be found by looking for a wikitext template that has the word "infobox" in its name.
                                                     If the user is not asking a question (e.g. "thank you", "thanks for the help"): Respond to it and ask the user if they have any further questions
                                                     Respond to greetings (e.g. "hi", "hello") with (in this exact order): A greeting, a brief description of yourself, and a question addressed to the user if they have a question or need assistance.

                                                     Please do not ask the user if they have any further questions, need further assistance, or the like.
                                                     Please do not ask the user if they have any further questions, need further assistance, or the like.
                                                     Please do not ask the user if they have any further questions, need further assistance, or the like.
                                                     Above all, be polite and helpful to the user. 
                                                     """));

        ClientResult<ChatCompletion>? clientResult = await chatClient.CompleteChatAsync(conversation);

        conversation.Add(new AssistantChatMessage(clientResult));

        // Maybe we shouldn't modify the conversation in place? Don't know if the caller minds that the UserChatMessage
        // was modified or not since all that really matters is the final response from the assistant.
        return conversation;
    }
}