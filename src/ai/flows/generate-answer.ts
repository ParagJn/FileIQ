// src/ai/flows/generate-answer.ts
'use server';

/**
 * @fileOverview A flow that takes a question and a set of documents, and returns a summarized answer based on the content of those documents.
 *
 * - generateAnswer - A function that takes a question and a set of documents, and returns a summarized answer.
 * - GenerateAnswerInput - The input type for the generateAnswer function.
 * - GenerateAnswerOutput - The return type for the generateAnswer function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateAnswerInputSchema = z.object({
  question: z.string().describe('The question to answer.'),
  documents: z.array(
    z.string().describe('A document to use to answer the question.')
  ).describe('The documents to use to answer the question.'),
});
export type GenerateAnswerInput = z.infer<typeof GenerateAnswerInputSchema>;

const GenerateAnswerOutputSchema = z.object({
  answer: z.string().describe('The summarized answer to the question.'),
});
export type GenerateAnswerOutput = z.infer<typeof GenerateAnswerOutputSchema>;

export async function generateAnswer(input: GenerateAnswerInput): Promise<GenerateAnswerOutput> {
  return generateAnswerFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateAnswerPrompt',
  input: {schema: GenerateAnswerInputSchema},
  output: {schema: GenerateAnswerOutputSchema},
  prompt: `You are a helpful assistant that answers questions based on the content of the provided documents.

  Question: {{{question}}}

  Documents:
  {{#each documents}}
  ---
  {{{this}}}
  {{/each}}
  ---

  Answer:`, 
});

const generateAnswerFlow = ai.defineFlow(
  {
    name: 'generateAnswerFlow',
    inputSchema: GenerateAnswerInputSchema,
    outputSchema: GenerateAnswerOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
