import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import RecipeListPage   from './pages/RecipeListPage'
import RecipeDetailPage from './pages/RecipeDetailPage'
import RecipeFormPage   from './pages/RecipeFormPage'
import FavoritesPage    from './pages/FavoritesPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/"                 element={<Navigate to="/recipes" replace />} />
        <Route path="/recipes"          element={<RecipeListPage />} />
        <Route path="/recipes/new"      element={<RecipeFormPage />} />
        <Route path="/recipes/:id"      element={<RecipeDetailPage />} />
        <Route path="/recipes/:id/edit" element={<RecipeFormPage />} />
        <Route path="/favorites"        element={<FavoritesPage />} />
      </Routes>
    </BrowserRouter>
  )
}
