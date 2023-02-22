int inputPin = 2;
int led = 13;
bool state = false;

void setup()
{
  Serial.begin(9600);
  pinMode(led,OUTPUT);
  pinMode(inputPin, INPUT);
  
  attachInterrupt(digitalPinToInterrupt(inputPin), swap, CHANGE);
}

void loop()
{
  delay(500);
}

void swap()
{
  state = !state;
  Serial.println("Motion Detected!");
  digitalWrite(LED_BUILTIN, state);
  Serial.println("Interrupted");
}
