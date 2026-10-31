from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

NUMERIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "duration": (1, 120),
    "credit_amount": (100, 50_000),
    "age": (18, 100),
    "installment_rate": (1, 4),
    "residence_since": (1, 4),
    "existing_credits": (1, 10),
}

CATEGORY_LEVELS: Dict[str, List[str]] = {
    "checking_account": ["<0 DM", "0-200 DM", ">=200 DM", "no account"],
    "credit_history": ["no credits / all paid", "all paid at this bank", "existing paid duly", "past delays", "critical account"],
    "purpose": ["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs",
                "education", "vacation", "retraining", "business", "others"],
    "savings_account": ["<100 DM", "100-500 DM", "500-1000 DM", ">=1000 DM", "unknown/none"],
    "other_debtors": ["none", "co-applicant", "guarantor"],
    "property": ["real estate", "savings agreement/life insurance", "car or other", "unknown/none"],
    "other_installment_plans": ["bank", "stores", "none"],
    "housing": ["rent", "own", "for free"],
    "telephone": ["none", "yes"],
    "employment_since": ["unemployed", "<1 year", "1-4 years", "4-7 years", ">=7 years"],
    "job": ["unemployed/unskilled non-resident", "unskilled resident", "skilled employee", "management/highly qualified"],
}

CheckingAccount = Literal["<0 DM", "0-200 DM", ">=200 DM", "no account"]
CreditHistory = Literal["no credits / all paid", "all paid at this bank", "existing paid duly", "past delays", "critical account"]
Purpose = Literal["car (new)", "car (used)", "furniture/equipment", "radio/television", "domestic appliances", "repairs",
                  "education", "vacation", "retraining", "business", "others"]
Savings = Literal["<100 DM", "100-500 DM", "500-1000 DM", ">=1000 DM", "unknown/none"]
OtherDebtors = Literal["none", "co-applicant", "guarantor"]
Property = Literal["real estate", "savings agreement/life insurance", "car or other", "unknown/none"]
InstallmentPlans = Literal["bank", "stores", "none"]
Housing = Literal["rent", "own", "for free"]
Telephone = Literal["none", "yes"]
Employment = Literal["unemployed", "<1 year", "1-4 years", "4-7 years", ">=7 years"]
Job = Literal["unemployed/unskilled non-resident", "unskilled resident", "skilled employee", "management/highly qualified"]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    duration: int = Field(ge=1, le=120, description="строк кредиту, місяці")
    credit_amount: Optional[float] = Field(default=None, ge=100, le=50_000, description="сума, DM; null — невідома")
    age: Optional[int] = Field(default=None, ge=18, le=100)
    installment_rate: int = Field(ge=1, le=4, description="внесок як частка доходу, шкала 1–4")
    residence_since: Optional[int] = Field(default=None, ge=1, le=4)
    existing_credits: int = Field(ge=1, le=10)
    checking_account: CheckingAccount
    credit_history: CreditHistory
    purpose: Optional[Purpose] = None
    savings_account: Optional[Savings] = None
    other_debtors: OtherDebtors
    property: Property
    other_installment_plans: InstallmentPlans
    housing: Housing
    telephone: Telephone
    employment_since: Optional[Employment] = None
    job: Optional[Job] = None

    @field_validator("credit_amount")
    @classmethod
    def amount_is_round_enough(cls, v):
        if v is not None and abs(v - round(v)) > 1e-6:
            raise ValueError("credit_amount має бути цілим числом DM")
        return v

    @model_validator(mode="after")
    def domain_consistency(self):
        if self.age is not None and self.employment_since == ">=7 years" and self.age < 23:
            raise ValueError("employment_since '>=7 years' неможливий при age < 23")
        if self.housing == "own" and self.property == "unknown/none" and self.other_debtors == "guarantor":
            raise ValueError("housing='own' суперечить property='unknown/none' за наявності поручителя")
        if self.credit_amount is not None and self.duration > 0 and self.credit_amount / self.duration > 10_000:
            raise ValueError("місячне навантаження понад 10 000 DM не відповідає продукту")
        return self


class Meta(BaseModel):
    model_version: str
    fallback_applied: bool
    ood_log_density: Optional[float] = None
    segment: Optional[Dict[str, object]] = None


class PredictionResponse(BaseModel):
    request_id: UUID
    score: float = Field(ge=0, le=1, description="P(bad) моделі; при fallback — довідково")
    label: int = Field(description="1 = bad (відмова / ручний розгляд), 0 = good; при fallback — рішення правила")
    uncertainty: float = Field(ge=0, description="ентропія прогнозного розподілу, нат")
    meta: Meta


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    items: List[PredictionRequest] = Field(min_length=1, max_length=256)


class BatchResponse(BaseModel):
    results: List[PredictionResponse]


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request_id: UUID = Field(strict=False, description="UUID4 з відповіді /v1/predict; рядок у JSON, тому strict=False на полі")
    y_true: Literal[0, 1]


class ExplainRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    request: PredictionRequest
    lime_samples: int = Field(default=1000, ge=200, le=2000)


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    request_id: Optional[str] = None
