const API_BASE_URL = 'http://localhost:8000/api';

export const generateArticle = async (topic: string, level: string): Promise<string> => {
  try {
    const response = await fetch(`${API_BASE_URL}/articles/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ topic, level }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.content;
    
  } catch (error) {
    console.error('Article generation error:', error);
    return '通信エラーが発生しました。バックエンドが起動しているか確認してください。';
  }
};