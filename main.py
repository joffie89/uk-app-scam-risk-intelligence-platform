from src.data_generation.customer_generator import CustomerGenerator

generator = CustomerGenerator()

for _ in range(5):
    customer = generator.generate_customer()
    print(customer)

  