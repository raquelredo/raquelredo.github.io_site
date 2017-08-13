---
layout: page
title: Portfolio
excerpt: "Projects done on Data Science"
search_omit: true
---
The following projects have been done on my own or while doing some DS Courses. You can access each individual coding just by clicking any of tem.

<ul class="post-list">
{% for post in site.categories.portfolio %}
  <li><article><a href="{{ site.url }}{{ post.url }}">{{ post.title }} <span class="entry-date"><time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%B %d, %Y" }}</time></span>{% if post.excerpt %} <span class="excerpt">{{ post.excerpt | remove: '\[ ... \]' | remove: '\( ... \)' | markdownify | strip_html | strip_newlines | escape_once }}</span>{% endif %}</a></article></li>
{% endfor %}
</ul>
