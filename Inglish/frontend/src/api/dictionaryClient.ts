const API_BASE_URL = 'http://localhost:8000/api';

export const translateText = async (text: string): Promise<string> => {
  try {
    // FastAPIの /api/translate エンドポイントにPOSTリクエストを送信
    const response = await fetch(`${API_BASE_URL}/translate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      // バックエンドの TranslationRequest スキーマに合わせて { text: "..." } という形にする
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    // 返ってきたJSONデータから翻訳結果を取り出す
    const data = await response.json();
    return data.translation;
    
  } catch (error) {
    console.error('Translation error:', error);
    return '通信エラーが発生しました。バックエンドが起動しているか確認してください。';
  }
};