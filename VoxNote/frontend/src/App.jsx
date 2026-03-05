import { useState } from 'react'
import axios from 'axios'

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  
  // コピーした直後かどうかを判定する状態
  const [isCopied, setIsCopied] = useState(false)

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('音声ファイルを選択してください')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    setIsCopied(false)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await axios.post('http://localhost:8000/api/transcribe', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setResult(response.data)
    } catch (err) {
      setError('エラーが発生しました。バックエンドが起動しているか確認してください。')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  // クリップボードにコピーする関数
  const handleCopy = () => {
    if (result && result.summary) {
      navigator.clipboard.writeText(result.summary)
      setIsCopied(true)
      // 2秒後に「コピーしました」の表示を元に戻す
      setTimeout(() => setIsCopied(false), 2000)
    }
  }

  return (
    <div style={{ 
      minHeight: '100vh', width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center',
      backgroundColor: '#ffffff', padding: '20px', boxSizing: 'border-box'
    }}>
      
      {/* くるくる回るアニメーション用のCSS（JSX内に直接定義） */}
      <style>{`
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
      `}</style>

      <div style={{ 
        width: '100%', maxWidth: '800px', backgroundColor: '#ffffff', padding: '50px', 
        borderRadius: '16px', boxShadow: '0 10px 30px rgba(0,0,0,0.05)', fontFamily: "'Helvetica Neue', Arial, sans-serif"
      }}>
        <h1 style={{ textAlign: 'center', color: '#2c3e50', marginTop: 0, fontSize: '2.5rem' }}>VoxNote</h1>
        <p style={{ textAlign: 'center', color: '#7f8c8d', marginBottom: '40px' }}>AI議事録作成SaaS</p>
        
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '20px' }}>
          <input 
            type="file" accept="audio/*" onChange={handleFileChange} 
            style={{ padding: '15px', border: '2px dashed #cbd5e1', borderRadius: '8px', width: '100%', maxWidth: '400px', cursor: 'pointer' }}
          />
          <button 
            type="submit" disabled={loading || !file} 
            style={{ 
              padding: '15px 40px', backgroundColor: loading ? '#94a3b8' : '#0ea5e9', color: 'white', 
              border: 'none', borderRadius: '8px', fontSize: '1.1rem', fontWeight: 'bold', 
              cursor: loading || !file ? 'not-allowed' : 'pointer', transition: 'background-color 0.3s',
              width: '100%', maxWidth: '400px', minHeight: '54px'
            }}
          >
            {loading ? (
              // ローディング中の表示（スピナー＋テキスト）
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '10px' }}>
                <div style={{ 
                  width: '20px', height: '20px', border: '3px solid rgba(255,255,255,0.3)', 
                  borderTop: '3px solid white', borderRadius: '50%', animation: 'spin 1s linear infinite' 
                }}></div>
                AI処理を実行中...
              </div>
            ) : (
              '文字起こし＆要約を実行'
            )}
          </button>
        </form>

        {error && <p style={{ color: '#ef4444', textAlign: 'center', fontWeight: 'bold', marginTop: '20px' }}>{error}</p>}

        {result && (
          <div style={{ marginTop: '50px', borderTop: '1px solid #e2e8f0', paddingTop: '30px' }}>
            <h2 style={{ textAlign: 'center', color: '#1e293b', marginBottom: '30px' }}>処理結果</h2>
            
            {/* 要約エリアのヘッダー（タイトルとコピーボタンを横並びに） */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
              <h3 style={{ color: '#0ea5e9', fontSize: '1.2rem', margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
                📝 要約 (Summary)
              </h3>
              <button 
                onClick={handleCopy}
                style={{
                  padding: '8px 16px', backgroundColor: isCopied ? '#10b981' : '#f8fafc', 
                  color: isCopied ? 'white' : '#475569', border: '1px solid #cbd5e1', borderRadius: '6px', 
                  cursor: 'pointer', fontWeight: 'bold', transition: 'all 0.2s', fontSize: '0.9rem'
                }}
              >
                {isCopied ? '✓ コピーしました' : '📋 コピーする'}
              </button>
            </div>
            
            <div style={{ backgroundColor: '#f8fafc', padding: '25px', borderRadius: '12px', whiteSpace: 'pre-wrap', border: '1px solid #e2e8f0', color: '#334155', lineHeight: '1.6' }}>
              {result.summary}
            </div>

            <h3 style={{ color: '#64748b', fontSize: '1.2rem', marginTop: '40px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
              🗣️ 文字起こし生データ (Transcription)
            </h3>
            <div style={{ backgroundColor: '#f1f5f9', padding: '25px', borderRadius: '12px', whiteSpace: 'pre-wrap', border: '1px solid #e2e8f0', color: '#475569', fontSize: '0.9rem', lineHeight: '1.6' }}>
              {result.transcription}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App