# SQLAlchemy ORM Quick Start

- source: `https://docs.sqlalchemy.org/en/20/orm/quickstart.html`
- curated_at: `2026-07-06 Asia/Shanghai`
- publisher: `SQLAlchemy official documentation`

SQLAlchemy 2.x ORM 使用 Declarative Base 定义映射类，`Mapped` 与 `mapped_column()` 描述 Python 属性和数据库列。`create_engine()` 创建 Engine；Engine 管理连接能力和连接池。`Base.metadata.create_all(engine)` 可以按 metadata 创建表。

ORM 的工作单元由 `Session` 管理。对象通过 `session.add()` 或 `add_all()` 加入，`commit()` flush 待处理变化并提交事务。官方示例推荐用 context manager 管理 Session 生命周期。查询使用 `select()` 构造语句，再由 `Session.scalars()`、`execute()` 或 `get()` 获取对象。

对象属性变化会被 Session 跟踪，在 flush 或 commit 时生成 UPDATE/INSERT/DELETE。关系可以用于 JOIN 和级联操作，但 lazy load 可能额外发出 SELECT，需要在真实查询中观察加载策略。

本项目通过 Engine + sessionmaker 管理 PostgreSQL 会话，ChatSession、ChatMessage、JudgeEvaluation 是 ORM 模型；RunRepository 则是独立 JSON 存储，不属于 ORM。

