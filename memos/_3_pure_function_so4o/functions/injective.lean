/-- 
  Checks if a function `f` is injective, 
  meaning that distinct inputs map to distinct outputs.
-/
def injective {α β : Type} (f : α → β) : Prop :=
  ∀ x y : α, f x = f y → x = y