---
  layout: "post"
  title: "Recognizing data types"
  excerpt: "Data has many shapes, colours and sizes. Can you see the difference?"
  categories: articles
  tags: [datascience,data, SQL, NoSQL, JASON, XML, storage]
  author: rachel
  comments: true
  share: true
  image:
    feature: banner2.jpg
    credit:
    creditlink:
  date: "2018-01-15 09:14"
  modified: "2018-01-15 09:14"
---


In the world of Data Science, there are many types of data. The kind of data you are going to work with will determine how you keep the data. There are different ways in which to save this data depending on them.
In general, we talk about

- **Structured data.** It has a specific and known order.
- **Semi-structured data.** Data has some structure but also accept some flexibility to change field names or create values.
- **Unstructured data.** They do not follow a specific scheme.

# Structured Data
It is the most common way in which we will find the data. The information is usually structured in columns with the variables, and the observations are the rows. It is the type of data that we commonly understand better as _data_. Whether we talk about financial statements in excel, bank statements or reports for the CEO, that has been the most common way to see data.
For this type of data, relational databases are the ones that work best. A structured language like [**SQL**](https://en.wikipedia.org/wiki/SQL), will combine perfectly to make calls to that data and give a result to the query. It does not mean that this data cannot be stored in NoSQL, but the performance it is much better.

# Semi-structured data
The data is rarely presented clean and structured. It is more usual to find the data in a formal way and in some field find intermingled data or a combination of different kinds.
For example, an email is considered semi-structured data and I find the best case to write it here. It has a part of structured data (sender and receiver's mail), but within the body of the message, we can have text, image, video, audio or a combination of any.

This is the most common type of data.
[**JavaScript Object Notation (JSON)**](https://en.wikipedia.org/wiki/JSON) or [**Extended Markup Language (XML)**](https://en.wikipedia.org/wiki/XML), are common terms when talking about the exchange of this type of data.

# Unstructured data
Some analysts estimate that 80% of the data created are unstructured. It makes some sense. Nowadays, the most normal thing is to create audio, a photo from your phone to send via Whatsapp or hang on your social networks. Even a PowerPoint would fit into this group of data.
As I have advanced before there is no scheme
New databases, such as the [**NoSQL**](https://en.wikipedia.org/wiki/NoSQL), allow the capture and storage of large files. Any grouping and data type can be stored in a NoSQL cluster.

The biggest challenge is what data need to be stored and what need to be deleted. In some cases it is cheaper to keep it as storage hard drives are very cheap. But this also increases the difficulty on knowing where to find the answer of your business question.

You will need to take decisions, and those decisions will absolutely have effect on the future analysis.
