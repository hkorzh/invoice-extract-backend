"""
The fixed extraction schema (frozen — see upwork.md). This IS the contract.

Lives in its own module because both the API layer (main.py, as a
response_model) and the Gemini layer (gemini_extractor.py, as a response_schema)
depend on it.
"""

from typing import Optional #, List

from pydantic import BaseModel, Field


class LineItem(BaseModel):
    description: Optional[str] = Field(None, description="What the line is for")
    quantity: Optional[float] = Field(None, description="Units billed")
    unit_price: Optional[float] = Field(None, description="Price per unit")
    amount: Optional[float] = Field(None, description="Line total")
    tax_percent: Optional[float] = Field(None, description="Percent of tax")


class Invoice(BaseModel):
    vendor: Optional[str] = Field(None, description="Company that issued the invoice")
    date: Optional[str] = Field(None, description="Invoice date exactly as printed")
    line_items: list[LineItem] = Field(default_factory=list)
    subtotal: Optional[float] = Field(None, description="Sum before tax")
    total_tax: Optional[float] = Field(None, description="Total tax")
    total: Optional[float] = Field(None, description="Total amount due")
    total_with_tax: Optional[float] = Field(None, description="Final amount due (with tax)")
