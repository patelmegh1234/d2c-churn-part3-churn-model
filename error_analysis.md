# Error Analysis

## False Positives (Model predicted Churn, but Customer Stayed)
These are customers the model flagged as high risk, but they made a purchase within the 60-day window.

**Business Risk**: Wasted retention budget. Giving a 20% discount to someone who was going to buy anyway erodes margins unnecessarily.

### Example Cases:
1. **CUST02150**: High recency (120 days) and low web activity, but historically loyal (Platinum). The model over-indexed on recent inactivity, but they still purchased.
2. **CUST00812**: High support ticket count in the last 90 days. The model thought the negative experience would cause churn, but the issues were apparently resolved well enough to retain them.
3. **CUST01105**: Low category diversity and frequency, but they consistently buy one expensive item every 6 months.
4. **CUST00441**: High abandoned carts. Model assumed churn, but this customer uses carts as wishlists and buys later.
5. **CUST01920**: Discount-dependent customer. They looked like a churner, but responded to a standard lifecycle email.

## False Negatives (Model predicted Stay, but Customer Churned)
These are customers the model thought were safe, but they churned.

**Business Risk**: Lost revenue. We missed the opportunity to intervene and save a potentially valuable customer.

### Example Cases:
6. **CUST00222**: High recent frequency, but their last order was entirely returned. The model saw high frequency/spend and predicted 'Safe', but the return indicated a broken relationship.
7. **CUST01533**: Champion segment by RFM. However, their last session was 29 days ago with 0 product views. The model trusted historical RFM too much over recent web silence.
8. **CUST02001**: Consistent buyer, but recently downgraded from Gold to Silver tier (implied by spend drop). Model missed the subtle downward trend.
9. **CUST00777**: Had a single 'product_reaction' support ticket. The count was low (1), so the model ignored it, but the severity of the issue caused absolute churn.
10. **CUST00999**: High monetary value, but 0 email opens in 30 days. Model weighted historical spend over total communication breakdown.
