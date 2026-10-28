-- SQLite 3.25+; the grain of the final table is ONE ORDER.
-- Aggregate each one-to-many table BEFORE joining it to orders.
CREATE TABLE order_mart AS
WITH ranked_items AS (
    SELECT i.*, p.product_category_name,
           COALESCE(t.product_category_name_english,
                    p.product_category_name, 'unknown') AS category,
           s.seller_state,
           ROW_NUMBER() OVER (
               PARTITION BY i.order_id
               ORDER BY i.price DESC, i.order_item_id ASC
           ) AS rn
    FROM items AS i
    LEFT JOIN products AS p USING (product_id)
    LEFT JOIN category_translation AS t USING (product_category_name)
    LEFT JOIN sellers AS s USING (seller_id)
), item_totals AS (
    SELECT order_id, COUNT(*) AS item_count,
           COUNT(DISTINCT seller_id) AS seller_count,
           SUM(price) AS goods_value, SUM(freight_value) AS freight_value,
           SUM(CASE WHEN price < 0 OR freight_value < 0
                         OR price IS NULL OR freight_value IS NULL
                    THEN 1 ELSE 0 END) AS invalid_item_count
    FROM items GROUP BY order_id
), payment_totals AS (
    SELECT order_id, SUM(payment_value) AS payment_value,
           COUNT(*) AS payment_records
    FROM payments GROUP BY order_id
), ranked_reviews AS (
    SELECT *, COUNT(*) OVER (PARTITION BY order_id) AS review_records,
           ROW_NUMBER() OVER (
               PARTITION BY order_id
               ORDER BY CASE WHEN review_answer_timestamp IS NULL THEN 1 ELSE 0 END,
                        review_answer_timestamp ASC, review_id ASC
           ) AS rn
    FROM reviews
    WHERE review_score BETWEEN 1 AND 5
)
SELECT o.*, c.customer_unique_id, c.customer_state,
       i.item_count, i.seller_count, i.goods_value, i.freight_value,
       i.invalid_item_count,
       p.payment_value, p.payment_records,
       r.review_score, r.review_answer_timestamp, r.review_records,
       d.seller_id AS main_seller_id, d.seller_state, d.category
FROM orders AS o
LEFT JOIN customers AS c USING (customer_id)
LEFT JOIN item_totals AS i USING (order_id)
LEFT JOIN payment_totals AS p USING (order_id)
LEFT JOIN ranked_reviews AS r ON o.order_id = r.order_id AND r.rn = 1
LEFT JOIN ranked_items AS d ON o.order_id = d.order_id AND d.rn = 1;

CREATE UNIQUE INDEX order_mart_order_pk ON order_mart(order_id);
CREATE INDEX order_mart_purchase_idx ON order_mart(order_purchase_timestamp);
