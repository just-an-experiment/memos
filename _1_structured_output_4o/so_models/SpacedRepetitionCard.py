# A Pydantic model representing a spaced repetition flashcard with review information.

from pydantic import BaseModel

class SpacedRepetitionCard(BaseModel):
    question: str
    """The question or prompt text displayed on the card."""

    answer: str
    """The answer or solution text for the question on the card."""

    last_reviewed: str
    """The date when the card was last reviewed, formatted as a string."""

    interval_days: int
    """The current review interval in days before the card will appear again."""

    easiness_factor: float
    """The easiness factor that influences the adjustment of the review interval."""

    repetitions: int
    """The number of times the card has been reviewed."""

    is_active: bool
    """Flag indicating whether the card is active in the spaced repetition cycle."""

export = SpacedRepetitionCard