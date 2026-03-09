import React, { useState } from 'react';
import { translateText } from '../../api/dictionaryClient'; 

export const FloatingWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [inputText, setInputText] = useState('');
  const [translation, setTranslation] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    
    setIsLoading(true);
    setTranslation('');
    
    try {
      const result = await translateText(inputText);
      setTranslation(result);
    } catch (error) {
      setTranslation('エラーが発生しました。');
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 bg-blue-600 text-white px-6 py-4 rounded-full shadow-lg hover:bg-blue-700 transition-all font-bold flex items-center gap-2"
      >
        <span className="text-xl">🔍</span> 翻訳
      </button>
    );
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 w-80 bg-white rounded-xl shadow-2xl border border-gray-200 overflow-hidden flex flex-col">
      <div className="bg-blue-600 text-white px-4 py-3 flex justify-between items-center">
        <h3 className="font-bold">Inglish 辞書</h3>
        <button onClick={() => setIsOpen(false)} className="text-white hover:text-gray-200 text-2xl leading-none">×</button>
      </div>
      <div className="p-4 flex flex-col gap-3">
        <textarea
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="わからない英単語やフレーズをペースト..."
          className="w-full h-24 p-2 border border-gray-300 rounded focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none text-sm"
        />
        <button
          onClick={handleTranslate}
          disabled={isLoading || !inputText.trim()}
          className="w-full bg-blue-600 text-white py-2 rounded font-medium hover:bg-blue-700 disabled:bg-blue-300 transition-colors flex justify-center items-center"
        >
          {isLoading ? '翻訳中...' : '翻訳する'}
        </button>
      </div>
      {translation && (
        <div className="p-4 bg-gray-50 border-t border-gray-100 min-h-[80px]">
          <p className="text-gray-800 text-sm whitespace-pre-wrap">{translation}</p>
        </div>
      )}
    </div>
  );
};