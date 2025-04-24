import { useState, useEffect, useRef } from 'react'
import reactLogo from './assets/react.svg'
import viteLogo from '/vite.svg'


function App() {
  const [count, setCount] = useState(0)
  const [query, setQuery] = useState('')
  const [responses, setResponses] = useState<string[]>([])
  const [isStreaming, setIsStreaming] = useState(true)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const responseEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when responses update
  useEffect(() => {
    if (responseEndRef.current) {
      responseEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [responses])

  // Handle regular (non-streaming) fetch
  const handleRegularFetch = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const response = await fetch('http://localhost:8000/api/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_text: query, stream: false }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`)
      }

      const data = await response.json()

      if (data.success) {
        setResponses(prev => [...prev, data.response])
      } else {
        setError(data.response || 'An unknown error occurred')
      }
    } catch (err) {
      setError(`Failed to fetch: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
    }
  }

  // Handle streaming fetch
  const handleStreamingFetch = async () => {
    try {
      setIsLoading(true)
      setError(null)

      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query_text: query, stream: true }),
      })

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`)
      }

      if (!response.body) {
        throw new Error('ReadableStream not supported in this browser')
      }

      // Initialize new response
      setResponses(prev => [...prev, ''])
      const responseIndex = responses.length

      // Process the stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()

      let done = false
      let accumulatedChunks = ''

      while (!done) {
        const { value, done: doneReading } = await reader.read()
        done = doneReading

        if (value) {
          const chunkText = decoder.decode(value, { stream: !done })
          accumulatedChunks += chunkText

          // Update the latest response with accumulated chunks
          setResponses(prev => {
            const updated = [...prev]
            updated[responseIndex] = accumulatedChunks
            return updated
          })
        }
      }
    } catch (err) {
      setError(`Failed to fetch: ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setIsLoading(false)
    }
  }

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!query.trim()) return

    if (isStreaming) {
      await handleStreamingFetch()
    } else {
      await handleRegularFetch()
    }
  }

  return (
    <div className="app-container">
      <header>
        <div className="logo-container">
          <a href="https://vite.dev" target="_blank" rel="noreferrer">
            <img src={viteLogo} className="logo" alt="Vite logo" />
          </a>
          <a href="https://react.dev" target="_blank" rel="noreferrer">
            <img src={reactLogo} className="logo react" alt="React logo" />
          </a>
        </div>
        <h1>Life-Block AI Assistant</h1>
      </header>

      <div className="chat-container">
        <div className="response-area">
          {responses.length === 0 ? (
            <div className="empty-state">Ask me anything...</div>
          ) : (
            responses.map((response, index) => (
              <div key={index} className="response-item">
                <div className="query-bubble">
                  {index === 0 ? query : `Follow-up query ${index}`}
                </div>
                <div className="response-bubble">
                  <pre>{response}</pre>
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="loading">
              <div className="dot-flashing"></div>
            </div>
          )}
          {error && <div className="error-message">{error}</div>}
          <div ref={responseEndRef} />
        </div>

        <form onSubmit={handleSubmit} className="query-form">
          <div className="input-container">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your query..."
              disabled={isLoading}
              className="query-input"
            />
            <button
              type="submit"
              disabled={isLoading || !query.trim()}
              className="submit-button"
            >
              Send
            </button>
          </div>

          <div className="streaming-toggle">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={isStreaming}
                onChange={() => setIsStreaming(!isStreaming)}
                disabled={isLoading}
              />
              Enable streaming responses
            </label>
            <div className="counter">
              <button onClick={() => setCount((count) => count + 1)} className="count-button">
                Count: {count}
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  )
}

export default App
