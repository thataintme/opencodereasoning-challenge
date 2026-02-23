import React, { useEffect, useState } from "react";
import supabase from "./supabaseClient";

type Example = {
  id: number;
  title: string;
  coding_question: string;
  user_code: string;
};

const App: React.FC = () => {
  const [codingQuestion, setCodingQuestion] = useState("");
  const [userCode, setUserCode] = useState("");
  const [examples, setExamples] = useState<Example[]>([]);
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  // Fetch example cards
  useEffect(() => {
    const fetchExamples = async () => {
      const { data, error } = await supabase
        .from("example_prompts")
        .select("*")
        .order("created_at", { ascending: true })
        .limit(10);

      if (error) {
        console.error("Error loading examples:", error);
      } else {
        setExamples(data || []);
      }
    };

    fetchExamples();
  }, []);

  const handleSubmit = async () => {
    if (!codingQuestion.trim() && !userCode.trim()) return;

    setLoading(true);
    setResponse("");

    try {
      const res = await fetch("/api/ask-ai", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          codingQuestion,
          userCode
        })
      });
      console.log(res)
      const data = await res.json();
      setResponse(data.answer);
    } catch (err) {
      console.error("AI request failed:", err);
      setResponse("Error contacting AI.");
    } finally {
      setLoading(false);
    }
  };

  const handleExampleClick = (example: Example) => {
    const hasInput =
      codingQuestion.trim() !== "" || userCode.trim() !== "";

    if (hasInput) {
      const confirmed = window.confirm(
        "This will replace your current input with the example. Continue?"
      );

      if (!confirmed) return;
    }

    setCodingQuestion(example.coding_question);
    setUserCode(example.user_code);
  };

  return (
    <div style={{ maxWidth: 1800, margin: "40px auto", padding: 20 }}>
      <h1>OpenCodeReasoning AI Gateway</h1>

      {/* Coding Question */}
      <h3>Coding Question</h3>
      <p>Copy the problem question and paste it here</p>
      <textarea
        rows={4}
        value={codingQuestion}
        onChange={(e) => setCodingQuestion(e.target.value)}
        style={{ width: "100%", marginBottom: 20 }}
      />

      {/* User Code */}
      <h3>User Code</h3>
      <p>Paste your buggy solution code here</p>
      <textarea
        rows={8}
        value={userCode}
        onChange={(e) => setUserCode(e.target.value)}
        style={{ width: "100%", fontFamily: "monospace" }}
      />

      <button
        onClick={handleSubmit}
        disabled={loading}
        style={{ marginTop: 20, padding: "10px 20px" }}
      >
        {loading ? "Thinking..." : "Submit"}
      </button>

      {/* Response */}
      {response && (
        <>
          <hr style={{ margin: "40px 0" }} />
          <h2>Model Response</h2>
          <div style={{ whiteSpace: "pre-wrap" }}>{response}</div>
        </>
      )}

      {/* Example Cards */}
      <hr style={{ margin: "40px 0" }} />
      <h2>Quick Examples</h2>

      <div style={{ display: "grid", gap: 10 }}>
        {examples.map((ex) => (
          <div
            key={ex.id}
            onClick={() => handleExampleClick(ex)}
            style={{
              padding: 15,
              border: "1px solid #ccc",
              borderRadius: 8,
              cursor: "pointer"
            }}
          >
            <strong>{ex.title}</strong>
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;