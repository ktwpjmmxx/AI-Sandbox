import { useState } from 'react';
import { FloatingWidget } from './components/Dictionary/FloatingWidget';
import { generateArticle } from './api/articleClient';

function App() {
  // 入力状態の管理
  const [topic, setTopic] = useState('Cloud computing');
  const [level, setLevel] = useState('中級者');
  
  // 記事データとローディング状態の管理
  const [article, setArticle] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);

  const handleGenerate = async () => {
    if (!topic.trim()) return;
    setIsLoading(true);
    setArticle(''); // 生成開始時に前の記事をクリア
    
    try {
      const content = await generateArticle(topic, level);
      setArticle(content);
    } catch (error) {
      setArticle('エラーが発生しました。');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-800 font-sans pb-24">
      <main className="max-w-3xl mx-auto p-8 pt-12">
        
        {/* ヘッダーと入力フォーム */}
        <div className="bg-white p-6 rounded-xl shadow-sm mb-8 border border-gray-100">
          <h1 className="text-2xl font-bold mb-4 text-blue-900">Inglish - AI Article Generator</h1>
          <div className="flex flex-col sm:flex-row gap-4">
            <input
              type="text"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              placeholder="テーマを入力 (例: AI, Space, History)"
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <select
              value={level}
              onChange={(e) => setLevel(e.target.value)}
              className="p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 bg-white"
            >
              <option value="初心者">初心者</option>
              <option value="中級者">中級者</option>
              <option value="上級者">上級者</option>
            </select>
            <button
              onClick={handleGenerate}
              disabled={isLoading}
              className="bg-blue-600 text-white px-6 py-3 rounded-lg font-bold hover:bg-blue-700 disabled:bg-blue-300 transition-colors whitespace-nowrap"
            >
              {isLoading ? '生成中...' : '記事を生成 ✨'}
            </button>
          </div>
        </div>

        {/* 記事の表示エリア */}
        {article && (
          <article className="bg-white p-8 rounded-xl shadow-sm border border-gray-100">
            <div className="prose prose-blue max-w-none space-y-4 text-lg leading-relaxed whitespace-pre-wrap">
              {article}
            </div>
          </article>
        )}

        {/* 初期状態のプレースホルダー */}
        {!article && !isLoading && (
          <div className="text-center text-gray-400 mt-12">
            テーマとレベルを指定して、「記事を生成」ボタンを押してください。
          </div>
        )}
      </main>
      
      {/* 辞書ウィジェット */}
      <FloatingWidget />
    </div>
  );
}

export default App;