# **App Name**: Document Vector Query

## Core Features:

- Folder Selection: Allow the user to select a local folder containing document files.
- Vector Generation: Upon folder selection, generate document embeddings for each file in the folder.
- Question Input: Present the user with a text input to ask questions about the documents.
- Contextual Answer Generation: Search the document vector store using the question from user, and return relevant information snippets as an answer. The LLM is a tool in determining what snippets best match the question.
- Answer Display: Display the answer on the screen in a clear and readable format.
- Data Simulation: Simulate data claims, records, and eligibility via structured JSON files for testing purposes.
- Step-by-Step UI: Display the application flow in stages using cards or steps, aided by Tailwind CSS.

## Style Guidelines:

- Primary color: Google Blue (#4285F4) as the primary Google brand color.
- Background color: Light gray (#FAFAFA), a very subtle shade to ensure readability and minimize distraction. It is derived from Google Blue (210 degrees in hue) at low saturation (10%) and high lightness.
- Accent color: Google Red (#EA4335), for a complementary contrast and to highlight key interactive elements.
- Body and headline font: 'Inter' sans-serif for a clean and modern aesthetic.
- Border styling of UI boxes/cards to use Google's brand colors (red, blue, yellow, green) using Tailwind CSS.
- Note: currently only Google Fonts are supported.