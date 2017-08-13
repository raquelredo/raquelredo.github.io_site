---
layout: page
title: Portfolio
excerpt: "Last logs about my changing career actions."
search_omit: true
---
The following projects have been done on my own or while doing some DS Courses. You can access each individual coding just by clicking any of tem.

<div class="grid__wrapper">
  {% for post in site.categories.portfolio %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
