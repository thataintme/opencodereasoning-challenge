import OpenAI from "openai";

const client = new OpenAI({
  baseURL: "https://router.huggingface.co/v1",
  apiKey: process.env.HF_TOKEN,
});

export default async (request) => {
  if (request.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const { codingQuestion, userCode } = await request.json();

  if (!codingQuestion && !userCode) {
    return new Response(JSON.stringify({ answer: "No input provided." }), {
      status: 400,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Using the exact same system prompt and user prompt structure as during training
  const system_prompt = `You are a Quality Assurance Engineer. Evaluate code formally and rigorously.
For each task, document your reasoning step by step, identify issues, and provide an objective final assessment in a structured format. Avoid informal language or assumptions; base all conclusions on evidence.
Always start with the <CoT>...</CoT> tags and provide your assessment. Follow that with the tags <propposed_fix>...</proposed_fix> with the correct code running python code. Next within <test_validation>...</test_validation> tags provide 3 assert statements that address the bug`

const user_prompt = `I am trying to solve this coding question and wrote this code, but there is something wrong with it. Can you find what the problem is?
The question:
${codingQuestion}
My code:
${userCode}`;

  const completion = await client.chat.completions.create({
    model: "santhosh-m/my-model-name", // TODO add correct gppo output here
    messages: [
        { role: "system", content: system_prompt },
        { role: "user", content: user_prompt }
    ],
    max_tokens: 4096,
  });

  const answer = completion.choices[0].message.content;

  return new Response(JSON.stringify({ answer }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
};

export const config = {
  path: "/api/ask-ai",
};