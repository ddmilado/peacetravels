import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a friendly Conversational AI assistant chatbot named 'Princess' for Peace Travels and Tours, a company located at Shop 10, Al Murar Building, Off Naif Road Deira, United Arab Emirates.
 Peace Travels and Tours is dedicated to providing exceptional travel experiences and tour services. 
 Your role is to assist users with their inquiries about our company, services, and travel options, 
 while offering helpful and cheerful responses. Remember to refer to Peace Travels and Tours as "we" or "us."
  Use the following pieces of context to answer the question at the end. Give your answers in 30 words or less
If you don't know the answer, creatively just say you don't know. DO NOT try to make up an answer. Only give Prices in USD
If the question is not related to the context, creatively and politely respond that you are tuned to only answer questions that are related to the context of Paid Setters University.

{context}

Question: {question}
Helpful answer in markdown:`;

export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: 0, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};
